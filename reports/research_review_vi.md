# Thẩm định nghiên cứu band-tilt và kế hoạch phát triển bài báo tạp chí

Ngày kiểm tra: 29/09/2026. Mã nguồn tại commit `ddc8a1278eda34150318fb05f6b307fc8f05cf04`.

**Kết luận:** đề tài có nền tảng hiện thực đủ tốt để phát triển tiếp, nhưng bằng chứng hiện tại mới là nghiên cứu khả thi trên một kịch bản. Chưa đủ để kết luận TuRBO ưu việt về hiệu năng mạng nói chung, chứng minh lợi ích riêng của phối hợp đa băng, hoặc bảo vệ một tuyên bố về tối ưu electrical tilt cho 5G/6G. Ưu tiên cho bài báo tạp chí là sửa sự tương ứng giữa bài toán, mô hình vật lý, hàm mục tiêu và thiết kế thực nghiệm; thêm MARL chưa phải việc cần làm trước.

## 1. Phạm vi và mức độ xác minh

Đã đọc README, báo cáo nghiên cứu, ADR về mục tiêu, cấu hình, luồng mô phỏng → KPI → tìm kiếm → công bố nghiệm → đánh giá; kiểm tra ba bộ `run.json`, `history.parquet`, `best_radio_map.npz` ngày 29/09/2026; đối chiếu một số nguồn nghiên cứu gốc và tài liệu thư viện. Đây là rà soát có mục tiêu, chưa phải tổng quan hệ thống toàn bộ văn liệu.

Đã chạy `.venv/Scripts/python.exe -m pytest -q`: **203 passed, 2 warnings, 34.72 s**. Hai cảnh báo là deprecation của `torch.jit.script`. Lần khởi chạy trong sandbox bị chặn quyền; chạy lại ngoài sandbox qua cơ chế cấp quyền đã thành công. Con số 205 trong README không khớp kết quả kiểm thử hiện tại.

Đã thực hiện kiểm tra Python chỉ đọc qua stdin:

- Tính lại toàn bộ chín KPI và J của ba nghiệm thắng từ bản đồ lưu trữ, với cấu hình của từng run và bảng UE hiện có. Sai khác tuyệt đối lớn nhất lần lượt là `7.71e-7`, `1.60e-6`, `1.39e-6` cho random, rule, TuRBO; đây là sai khác số học nhỏ, không phải phép đánh giá độ đúng vật lý.
- Xác nhận random và TuRBO có cùng cấu hình ban đầu và 16 điểm khởi tạo Sobol, so trực tiếp 36 cột góc nghiêng.
- Đếm cấu hình phân biệt: random 145/145; rule **48/111**; TuRBO 145/145.
- Xác nhận phản ví dụ: utility ba băng `(1, 0.1, 0)` cho J tại ô bằng `0.9181818`; làm băng thứ hai mất phủ sóng cho `(1, 0, 0)` thì J tăng thành `1`.
- Xác nhận `seed=1` thay đổi định danh scenario; `optim.seed=1` giữ nguyên scenario.

Không chạy lại ray tracing, không huấn luyện/tối ưu thêm, không có số đo mạng thật và chưa kiểm chứng hội tụ của bộ giải. Không sửa mã nguồn hoặc kết quả nghiên cứu gốc trong đợt thẩm định này.

## 2. Kết quả hiện tại thực sự nói gì?

Nguồn: `reports/tables/04_evaluation/kpi_scoreboard.csv`, `method_cost.csv`, cùng metadata và bản đồ đã tính lại. Mỗi phương pháp chỉ có một search seed: 42.

| Chỉ tiêu | Ban đầu | Sobol | Quét theo luật | TuRBO |
|---|---:|---:|---:|---:|
| J, lớn hơn tốt hơn | 0.665142 | 0.691112 | 0.696296 | **0.706100** |
| Lỗ phủ sóng, % diện tích | 11.249 | 10.825 | **10.442** | 10.784 |
| Phủ sóng yếu, % diện tích | 30.692 | 28.635 | **25.353** | 26.249 |
| Chồng phủ đồng băng, % diện tích | 31.640 | 32.538 | 31.806 | **31.019** |
| Số láng giềng chồng phủ/ô được phủ | 0.9625 | 1.0303 | 1.1866 | 0.9841 |
| RSRP p50, dBm | −84.107 | −82.718 | **−80.797** | −81.513 |
| RSRP p05, dBm | −108.564 | −107.982 | **−106.968** | −107.315 |
| SINR p50, dB | 14.264 | 15.033 | 14.870 | **16.404** |
| SINR p05, dB | −0.875 | −1.096 | −1.500 | **−0.724** |
| UE report không được phục vụ, % | 21.741 | 21.186 | **21.087** | 21.235 |
| Thời gian toàn run, phút | — | 21.80 | **10.85** | 39.52 |
| Số KPI cải thiện / 9 | — | 6 | 6 | 8 |

TuRBO tăng J **6.16% tương đối** so với ban đầu. Điều đó không có nghĩa diện tích phủ sóng hoặc throughput tăng 6.16%. Lỗ phủ giảm **0.465 điểm phần trăm**, tương đương giảm tương đối 4.13% số ô lỗ phủ. UE failure giảm **0.506 điểm phần trăm**, tương ứng thêm **51 UE reports** được phục vụ trong 10,087 reports; không được gọi là 51 người dùng duy nhất hay 51 kết nối liên tục.

TuRBO hơn rule `0.009804` về J, khoảng 1.41% tương đối, và tốn khoảng 3.64 lần thời gian toàn run đã ghi nhận. Rule phục vụ thêm 66 reports so với ban đầu, hơn TuRBO 15 reports. Các số này mô tả đúng một thực nghiệm, chưa có khoảng tin cậy cho khác biệt giữa thuật toán.

**Kết luận bảo vệ được:** trong kịch bản đã đo, TuRBO tìm được J cao nhất và một đánh đổi tốt hơn về chồng phủ/SINR; rule tốt hơn về lỗ phủ, RSRP và admission. Không có phương pháp thống trị toàn bộ KPI. Đếm “8/9 KPI tốt hơn” cũng không tạo ra tám bằng chứng độc lập, vì nhiều KPI cùng được suy ra từ RSRP.

Điểm mạnh hiện có là dùng chung evaluator, lưu history/configuration, ngân sách random–TuRBO bằng nhau và có xác minh KPI từ bản đồ. Những điều này đáng giữ lại.

## 3. Những vấn đề cần giải quyết trước khi nộp tạp chí

### A. Góc nghiêng được hiện thực không tương ứng với tên electrical tilt

`src/simulation/transmitter.py:212` đặt góc vào `Transmitter.orientation`, cụ thể thành phần pitch tại dòng 233. `src/simulation/radio.py::solve_band` không truyền vector precoding phụ thuộc tilt. Mã thư viện đã cài đặt dùng vector mặc định đều, được chuẩn hóa theo số phần tử anten.

Đây là phép xoay thiết bị/mảng và giản đồ phần tử, có bản chất hình học gần với mechanical tilt. Electrical downtilt thường đòi hỏi thay đổi kích thích/pha của mảng trong khi giữ hướng vật lý của panel cố định. Tài liệu Sionna phân biệt [orientation của thiết bị](https://nvlabs.github.io/sionna/rt/api/radio_devices.html) với [precoding của mảng trong radio-map solver](https://nvlabs.github.io/sionna/rt/api/radio_map_solvers.html).

Hai hướng hợp lệ:

1. Giữ hiện thực, gọi đúng là tối ưu hướng/góc nghiêng hình học của panel theo băng và nêu giả định các panel có thể điều chỉnh độc lập.
2. Nếu mục tiêu là RET/electrical tilt đa băng, cố định orientation cơ khí, điều khiển pha/precoder hoặc dùng giản đồ anten được hiệu chuẩn theo electrical tilt. Kiểm tra trước trên scene LOS đơn giản và so giản đồ phương đứng ở vài góc.

Không khẳng định thay đổi tên là đủ để mô hình tương đương phần cứng thực. Với panel đa băng dùng chung kết cấu, 36 góc cơ khí độc lập cũng cần giải thích tính khả thi. Nếu chuyển sang electrical tilt, phải chạy lại kết quả chính. Chú ý `uv.lock` hiện khóa Sionna-RT 2.0.1, trong khi tài liệu web đang ở 2.1.0; quy ước pha phải kiểm tra theo đúng phiên bản chạy.

### B. Lập luận về chồng phủ đa băng đang có sai lệch vật lý

Phần Introduction và Recommendations của báo cáo gắn phủ trùng giữa các băng với nhiễu, rồi đề nghị thêm inter-band interference. Nhưng hiện thực giải từng băng riêng; SINR chỉ cộng các transmitter đồng băng. Đây là giả định hợp lệ cho các sóng mang tách biệt nếu nêu rõ bỏ qua rò phổ và phi tuyến RF.

Không nên thêm công suất 700/1800/2600 MHz vào cùng mẫu số SINR chỉ vì cùng phủ một vị trí. Nếu nghiên cứu nhiễu xuyên băng, cần mô hình rò phổ, lọc thu, blocking hoặc intermodulation cụ thể. Chồng phủ không gian khác băng tự nó không phải bằng chứng lãng phí: có thể cung cấp lựa chọn phục vụ, dung lượng hoặc dự phòng.

**Nên viết lại động cơ:** phối hợp các lớp phủ để phân bổ phục vụ và tài nguyên tốt hơn, đồng thời quản lý nhiễu giữa các cell cùng băng. Đây cũng là giả định trực tiếp của [SINR radio map Sionna](https://nvlabs.github.io/sionna/rt/api/radio_map_solvers.html) và mã `radio_map.py::sinr` đã kiểm tra tại máy.

### C. J có thể thưởng cho việc làm suy giảm một lớp phủ

Tại một ô, đặt `C = Σu²/Σu`. Sau khi bỏ một băng có utility `u`, điểm tăng khi `0 < u < C`, với điều kiện vẫn còn tổng utility dương. Ví dụ vừa kiểm chứng bằng chính hàm `effective_coverage`: `(1, 0.1, 0) → (1, 0, 0)` làm điểm tăng từ 0.9182 lên 1.

Thậm chí `∂C/∂u = (2u − C)/Σu`: khi `u < C/2`, cải thiện một băng yếu có thể làm điểm tổng giảm. Vấn đề không chỉ nằm ở thao tác xóa hẳn băng.

Trên bản đồ TuRBO đã lưu, phép tính kiểm tra cho thấy khoảng **49% ô** có ít nhất một utility dương nhỏ hơn C; chênh lệch trung bình giữa `max(u)` và C khoảng **0.05428**. Đây là phân tích đại số trên bản đồ, không phải một cấu hình tilt khả thi hoặc bằng chứng optimizer đã khai thác hết mức đó. Không dùng so sánh 0.05428 với ΔJ để khẳng định toàn bộ cải thiện là giả.

Hệ quả khi viết bài:

- Gọi J là **chỉ số tiện ích phủ sóng tự thiết kế**, không đồng nhất nó với phần trăm diện tích phủ, throughput, hoặc xác suất được phục vụ.
- “Không có tham số” là quá mạnh: J vẫn phụ thuộc −120 dBm, −90 dBm, 6 dB và dạng penalty mũ được chọn. Cách nói đúng là “không thêm trọng số ngoài các ngưỡng KPI đã chọn”.
- Đưa tính không đơn điệu vào ablation và kiểm tra trade-off. Việc bỏ một băng yếu độc lập ở từng ô chỉ là phản thực tế toán học; thực nghiệm khả thi phải thay góc của cell-band rồi giải lại toàn bộ bản đồ.
- Tương quan cao giữa J và KPI RSRP cùng thành phần không xác nhận utility mạng. Cần kiểm định trên chỉ tiêu dịch vụ và môi trường chưa dùng để thiết kế J.

### D. Chưa chứng minh lợi ích riêng của phối hợp đa băng

Rule hiện dùng một góc chung cho toàn bộ 12 cell trên từng băng: chỉ ba bậc tự do. TuRBO có 36 biến. So sánh đó đồng thời thay đổi thuật toán và mức tự do, nên không tách được lợi ích của “phối hợp đa băng” khỏi lợi ích của “cho phép mỗi cell có góc riêng”.

Cần ít nhất hai đối chứng bổ sung:

- Tối ưu độc lập từng băng, mỗi băng vẫn có 12 góc, sau đó ghép nghiệm và đánh giá trên cùng chỉ tiêu dịch vụ đa băng.
- Một thuật toán tìm kiếm đơn giản có đủ 36 biến, ví dụ coordinate/pattern search. Nếu cần thêm đối chứng tối ưu hộp đen, dùng differential evolution có sẵn trong SciPy và kiểm soát ngân sách hàm gọi.

Giữ rule ba biến như một baseline vận hành dễ diễn giải, không gọi nó là đối chứng đầy đủ cho tối ưu độc lập từng băng. Đối chứng phải được tính công bằng cả số lượt gọi bản đồ từng băng, không chỉ số vector 36 chiều được đánh giá.

### E. “Một tuần UE” chưa làm bài toán trở thành tối ưu động hoặc theo nhu cầu

`src/optim/objective.py:129` chỉ đọc RSRP. Bảng UE tham gia `evaluate_kpis` để tính failure, nhưng không quyết định xếp hạng nghiệm. Các interval là các lần lấy mẫu vị trí; không có điều khiển tilt theo thời gian, handover hoặc trạng thái phiên liên tục.

10–20 UE mỗi interval là tổng trên mạng 12 cell, không phải mỗi cell. Tại cấu hình ban đầu, 2,142 trong 2,193 reports thất bại nằm ở ô lỗ phủ; chỉ 51 reports thất bại trên ô được phủ. Vì vậy thực nghiệm hiện chủ yếu kiểm tra coverage, chưa tạo sức ép đủ rộng để bảo vệ luận điểm cân bằng tải/dung lượng.

Demand dựa trên PRB yêu cầu bỏ qua UE không có cell khả dụng vì nhu cầu của chúng là NaN. “Demand trong lỗ phủ bằng 0” không có nghĩa nhu cầu ngoài vùng phủ bằng 0. Khi đánh giá theo nhu cầu, dùng **số UE reports hoặc lưu lượng yêu cầu trước admission**, độc lập với việc mạng phục vụ được hay không.

### F. Độ đúng mô hình cần được kiểm chứng riêng với độ đúng phần mềm

Các vấn đề đã thấy trong cấu hình và mã:

- Diffraction và diffuse reflection đang tắt. Không tìm thấy ray ở cấu hình hiện tại chưa chứng minh không thể có đường truyền vật lý. Cần kiểm tra hội tụ số ray, độ sâu và propagation mechanisms trước khi gọi một vùng là “không thể cứu bằng tilt”.
- `materials.evaluate` ngoại suy hệ số ITU ngoài dải công bố. Với vật liệu nào nằm ngoài dải ở 700 MHz, phải liệt kê và phân tích độ nhạy; không mô tả như mọi thông số đều được ITU xác nhận trong dải. Đối chiếu [ITU-R P.2040](https://www.itu.int/rec/R-REC-P.2040).
- SINR giả định các transmitter đồng băng luôn phát đầy đủ, trong khi admission tính tải từng cell. Đây có thể là giả định bảo thủ cho nhiễu, nhưng phải nhất quán trong diễn giải.
- Noise chưa có receiver noise figure; Shannon efficiency chưa có trần MCS, overhead hoặc hiệu chuẩn link adaptation. Số lớp MIMO cố định một không nhất thiết tạo sai số theo hướng lạc quan. Không nên gọi tổng hợp mọi xấp xỉ là upper bound của mạng NR thực.
- `power_rs=4.85 dBm` phải có bảng power budget: công suất tham chiếu trên RE, chuẩn hóa precoder, array gain và công suất tổng theo băng. Công thức RSS/SINR đã dùng nhất quán trên cùng cơ sở RE, nhưng chưa mô phỏng đầy đủ đo lường SSB/CSI-RS hay data scheduling của NR.
- Mảng 8×8, khoảng cách 0.5 bước sóng trên cả ba băng tương ứng các kích thước vật lý khác nhau. Có thể dùng làm giả định nghiên cứu, cần mô tả đúng phần cứng tương đương.
- Miền tính J là toàn bộ lưới chữ nhật, còn UE được lấy mẫu trên vùng đủ điều kiện. Cần bản đồ mặt nạ vùng phục vụ cố định và độ nhạy với biên miền, đặc biệt các ô trong footprint công trình hoặc xa cụm trạm. Không được loại ô dựa trên nghiệm thắng.

Độ cao trạm 25 m, UE 1.5 m và tilt ban đầu 8–12° cũng cần được giải thích bằng thiết kế vùng phục vụ. Xấp xỉ LOS phẳng cho giao điểm trục chính là `(25−1.5)/tan(tilt)`, khoảng 111–167 m, ngắn hơn nhiều khoảng cách liên trạm 1,732 m. Đây không phải bán kính phủ, nhưng là lý do phải kiểm tra baseline thiết kế tốt hơn trước khi nhấn mạnh gain so với cấu hình đầu.

### G. Sai sót báo cáo và tái lập có thể sửa với thay đổi nhỏ

| Mức ưu tiên | Vị trí | Phát hiện | Điều chỉnh đề xuất |
|---|---|---|---|
| Cao | `src/evaluation/compare.py:106`, `:678` | Tính % cải thiện cho cả số đo dBm/dB | RSRP/SINR báo ΔdB; tỷ lệ báo điểm phần trăm, có thể thêm % tương đối. Nếu muốn tỉ số công suất, chuyển sang tuyến tính và nói rõ đại lượng. |
| Cao | `src/evaluation/compare.py:253` | Đếm J vào “KPIs improved” | Lọc theo `KPI_NAMES`; hiện CSV ghi 9/1 cho TuRBO, đúng trên 9 KPI phải là 8/1. |
| Cao | `src/evaluation/runs.py:176`, `:207`; `src/simulation/scenario.py:35` | Chọn latest theo method/seed; định danh scenario dựa config/path, chưa hash nội dung mesh, objective và dữ liệu | Định danh experiment bao gồm scene/data/code/objective; nhóm theo experiment trước khi chọn run, không xóa bản cũ để tránh trộn. |
| Cao | `configs/config.yaml`, `configs/optim/base.yaml`, `configs/simulation.yaml` | `seed` gốc điều khiển cả scenario và optimizer | Search-seed sweep phải override `optim.seed`; solver-seed và scenario-seed là các trục khác nhau. `task sweep -- seed=...` không chỉ đổi optimizer và có thể bị manifest hiện tại từ chối. |
| Vừa | `src/optim/methods/turbo/search.py:221` | Dùng `SingleTaskGP` mặc định; thư viện hiện tại chọn RBF | Ghi rõ kernel, priors, noise model, normalization, phiên bản; không gọi là tái hiện nguyên xi tutorial Matérn. |
| Vừa | `src/optim/methods/rule/search.py:75` | 111 lượt nhưng 48 cấu hình phân biệt | Cache theo cấu hình và điều kiện evaluator cố định, hoặc dừng sau một vòng không cải thiện; phân biệt re-evaluation có chủ đích với lặp lãng phí. |
| Vừa | `src/kpi/quality.py`, `src/kpi/capacity.py::serving_sinr` | SINR p05 hiện là strongest-RSRP server trên các ô được phủ, khác association thực của UE | Giữ KPI này nhưng ghi đúng tên; thêm serving-SINR/throughput của UE và outage trên tập UE cố định. |
| Vừa | `configs/config.yaml` | MLflow đang `enabled: false`, README nói đang log mọi stage | Đồng bộ tuyên bố với config, lưu provenance trực tiếp trong run kể cả khi không dùng MLflow. |

Các percentile chỉ tính trên các ô được phủ của từng nghiệm có thể tăng do mất các ô yếu. Báo cả phân bố trên miền đánh giá cố định và outage; không coi p05 có điều kiện là bằng chứng duy nhất về cell edge.

Việc làm tròn J đến 1e−6 có thể tăng khả năng lặp lại trong môi trường hiện tại, nhưng không đo bất định Monte Carlo và không bảo đảm tái lập trên mọi máy. Giữ giá trị chưa làm tròn để kiểm tra độ nhạy và lưu môi trường chạy.

## 4. Định vị đóng góp và hướng bài báo nên chọn

“Dùng Bayesian optimization để tối ưu tilt” đã có tiền lệ. [Dreifuerst và cộng sự, ICASSP 2021](https://arxiv.org/abs/2010.13710) nghiên cứu tối ưu power/downtilt bằng BO và RL. [Tekgul và cộng sự](https://arxiv.org/abs/2210.15732) tối ưu cấu hình anten cho coverage/capacity cả UL và DL bằng phương pháp dựa trên BO và differential evolution. [TuRBO của Eriksson và cộng sự](https://arxiv.org/abs/1910.01739) là thuật toán đã công bố, không phải đóng góp thuật toán mới của đề tài này.

Các nguồn đó đủ để bác bỏ cách định vị novelty quá rộng; chưa đủ để kết luận phối hợp đa băng cụ thể của bạn không mới. Cần lập bảng related work đọc toàn văn theo các trục: band/cell coupling, association, traffic/load, mô hình anten, simulator/calibration, số biến, ngân sách và uncertainty. Không so trực tiếp phần trăm gain giữa các bài dùng scenario khác nhau.

**Hướng khuyến nghị:** nghiên cứu tối ưu tilt đa băng theo chất lượng dịch vụ dưới ngân sách mô phỏng hạn chế, có kiểm định bền vững trước thay đổi môi trường. TuRBO là công cụ tìm kiếm; đóng góp nằm ở formulation có ý nghĩa vật lý, cách phối hợp các lớp và bằng chứng định lượng có đối chứng.

Tên làm việc: **“Service-Aware Multi-Band Antenna Tilt Coordination under Limited Ray-Tracing Budgets”**. Đây là tên cho hướng mở rộng, chưa phải mô tả đúng kết quả hiện tại. Chỉ thêm “electrical” sau khi đã hiện thực và kiểm chứng electrical tilt; chỉ thêm “robust” sau khi thực nghiệm chứng minh được robustness. Bỏ “6G” khỏi tuyên bố chính vì mô hình hiện tại chưa có thành phần đặc thù 6G.

Ba câu hỏi nghiên cứu:

1. Với cùng quyền điều chỉnh góc, phối hợp đa băng có cải thiện dịch vụ so với tối ưu độc lập từng băng không?
2. Với cùng ngân sách đánh giá và cùng thời gian, trust-region BO có hiệu quả hơn các phương pháp đơn giản không?
3. Nghiệm tìm được có giữ lợi ích trên traffic, solver seeds và sai lệch mô hình chưa dùng để chọn nghiệm không?

### Chốt hàm mục tiêu theo dịch vụ

Một lựa chọn tối giản để thí nghiệm là:

`F(theta) = mean_s [ Σ_u w_us · 1{UE u được admission đạt demand trong scenario s} / Σ_u w_us ]`.

`w_us` lấy từ nhu cầu ngoại sinh hoặc bằng 1, không suy từ PRB đã được mạng chấp nhận. Nếu tất cả UE có cùng demand thì w=1 là đủ. Tối ưu F kèm ràng buộc không làm hole rate vượt cấu hình tham chiếu quá một dung sai định trước; nếu giữ vai trò lớp 700 MHz thì viết thành yêu cầu coverage cụ thể cho lớp đó.

Dung sai và ngưỡng phải chốt từ yêu cầu nghiên cứu/vận hành trước khi xem test, không chọn để TuRBO thắng. Có thể bắt đầu bằng chọn nghiệm khả thi trong tập điểm đã đánh giá; đây là luật lựa chọn có ràng buộc, **chưa tương đương** một acquisition tối ưu ràng buộc.

Nếu admission tạo plateau quá lớn, thử utility `mean(min(achieved_rate/demand, 1))`, nhưng phải bổ sung mô hình achieved rate thực sự. Mã hiện tại tính nhu cầu PRB và admit/reject, không đủ để tự tuyên bố đã tối ưu delivered throughput. Giữ J hiện tại làm baseline/ablation, không thay âm thầm rồi gộp các run.

Không bắt buộc thiết kế thuật toán mới. Nếu lựa chọn tạp chí yêu cầu đóng góp thuật toán, cần một cải tiến có cơ sở riêng và ablation chứng minh phần cải tiến; chỉ đổi số chiều perturbation từ 20 xuống 5 chưa đủ.

## 5. Thiết kế thực nghiệm cho bản tạp chí

Các số lượng dưới đây là kế hoạch khởi đầu đề xuất, không phải tiêu chuẩn mặc định bảo đảm có ý nghĩa thống kê.

| Khối | Thí nghiệm cần làm | Câu hỏi được giải quyết |
|---|---|---|
| Kiểm chứng vật lý | Scene LOS nhỏ; các tilt chuẩn; giản đồ đứng/ngang; power và noise budget; electrical vs geometric nếu cần | Biến điều khiển có đúng thứ đang tuyên bố? |
| Hội tụ bộ giải | Ban đầu và 2–3 nghiệm đại diện; nhiều solver seeds; tăng số ray, so grid/depth; bật diffraction ở tập con | Ranking có phải do nhiễu/tổn thất do truncation? |
| Đối chứng | Incumbent đã hiệu chỉnh; rule 3D; coordinate search 36D; Sobol; TuRBO; tối ưu độc lập từng băng | Lợi ích do thuật toán, độ tự do hay coupling? |
| Search seeds | Khởi đầu 10 seeds cho phương pháp ngẫu nhiên, cùng initial design theo cặp; điều chỉnh cỡ mẫu sau pilot và mục tiêu precision đã chốt | Độ ổn định và hiệu quả tìm kiếm |
| Kịch bản | Ít nhất 3 layout/scene khác nhau nếu muốn khái quát về địa hình; load thấp/vừa/cao và hotspot dịch chuyển | Khái quát ngoài đúng scene hiện tại |
| Phân tách dữ liệu | Bộ development để chốt objective/hyperparameters; tập traffic/channel test riêng; benchmark môi trường mới có giao thức cố định | Tránh chỉnh thiết kế theo kết quả test |
| Ablation objective | J hiện tại; max utility; trung bình với mẫu số band cố định; objective dịch vụ đề xuất | Non-monotonicity và proxy mismatch quan trọng đến đâu? |
| Ablation coupling | Cùng thuật toán, joint vs per-band độc lập vs shared-tilt, cùng ngân sách quy đổi | Giá trị của đa băng |
| Độ nhạy | Ngưỡng RSRP/overlap, admission 0.8, band preference, noise figure, công suất, vật liệu, UE order | Kết luận có phụ thuộc lựa chọn thuận lợi? |
| Khả thi phần cứng | Lượng tử hóa góc theo thiết bị; giới hạn thay đổi; sai lệch góc; đánh giá lại sau lượng tử hóa | Nghiệm số có dùng được với cơ cấu thực? |
| Độ mở rộng | Tăng số cell/biến trên một vài cỡ phù hợp tài nguyên | Chi phí và gain khi bài toán lớn hơn |

**Phân biệt ba loại lặp:** search seed đo bất định thuật toán; solver seed đo nhiễu đánh giá cùng nghiệm; scenario đo độ khái quát. Chạy lại cùng seed để có cùng kết quả không thay thế ba phép kiểm tra này.

Đối với tối ưu offline không học một policy chuyển giao, có hai phép thử khác nhau: (a) giữ theta cố định và thử traffic/channel mới; (b) chạy lại thuật toán theo giao thức đã đóng băng trên layout mới. Phép (b) đo độ khái quát của phương pháp tối ưu, không phải khả năng chuyển một vector tilt 36 chiều sang mọi mạng.

Với dữ liệu một tuần có cấu trúc theo thời gian, chia development/test theo các khối hoặc scenario; không chia ngẫu nhiên từng UE row rồi coi đó là các thực nghiệm độc lập. Các ô không gian cũng tương quan. Khoảng tin cậy so thuật toán phải dùng run/scenario làm đơn vị phù hợp, không lấy 101,060 ô hay 10,087 reports làm số lần lặp độc lập của optimizer.

**Phân tích thống kê và công bằng:** báo chênh lệch theo cặp, khoảng tin cậy 95%, effect size thực dụng và số scenario có vi phạm ràng buộc. Với nhiều seeds trên nhiều scene, tổng hợp theo scene hoặc bootstrap phân cấp; không gộp mọi quan sát như độc lập. Nếu dùng Wilcoxon signed-rank thì cần lưu ý giả định đối xứng của phân bố chênh lệch; không xem p-value là thay thế effect size. Chọn trước chỉ tiêu chính và xử lý multiplicity nếu kiểm định nhiều KPI.

So sánh cả `best-so-far` theo số oracle calls và theo thời gian thực trên cùng máy, warm-up và thứ tự chạy cân bằng. Batch ba điểm hiện được evaluator giải tuần tự; không gọi đó là tăng tốc đánh giá song song. Kể cả lượt re-trace cuối cùng để lưu nghiệm cũng tiêu tốn tính toán, dù không nằm trong history tìm kiếm.

Không chạy lại rule xác định 10 lần trên cùng môi trường chỉ để tạo mười “seed thuật toán”; so nó với các nghiệm ngẫu nhiên và lặp theo solver/scenario khi cần. Cùng seed solver giúp tạo điều kiện so sánh theo cặp, không bảo đảm nhiễu triệt tiêu hoàn toàn.

Nếu có dữ liệu đo, dùng tập hiệu chuẩn vật liệu/path loss và tập kiểm tra độc lập, báo bias/RMSE/quantile sai số. Nếu không có, cung cấp synthetic scene tái tạo được, kiểm chứng bộ giải và sensitivity; giới hạn kết luận ở mô phỏng. Không cần hứa triển khai mạng thật để viết một bài nghiên cứu mô phỏng nghiêm túc.

## 6. Tối ưu hiện thực có giá trị ngay

1. **Loại đánh giá trùng trước.** 63/111 lượt rule là cấu hình đã gặp. Cache chỉ dùng lại khi vector, scenario, objective và solver settings/seed cùng nhau; phép đo lặp để ước lượng nhiễu phải được chủ động bỏ qua cache. Báo riêng unique configurations và simulator calls sau thay đổi.
2. **Đo thời gian theo khâu.** TuRBO ghi 39.52 phút toàn run nhưng khoảng 8.14 phút ray tracing trong history. Phần chênh không được quy hết cho GP: còn KPI/admission, setup, fit, sampling và lưu bản đồ. Thêm timer nhỏ ở đúng các khâu trước khi tối ưu.
3. **Tách scoring rẻ và hậu kiểm đắt nếu vẫn giữ J.** Hiện mỗi candidate tính admission của mọi UE dù J không đọc kết quả. Có thể đo J trên mọi điểm và chỉ tính dịch vụ cho shortlist hoặc các mốc cố định, khi giao thức nghiên cứu không cần KPI dịch vụ của mọi candidate. Nếu dùng mục tiêu dịch vụ mới, không áp dụng lược bỏ này.
4. **Tái sử dụng bản đồ băng không đổi.** Rule chỉ đổi một băng mỗi bước; hiện solver giải lại cả ba. Với mô hình các băng độc lập hiện tại, có thể cache hai bản đồ còn lại. Đếm đúng số band-solves và xác minh tương đương trước khi công bố tăng tốc. Tái sử dụng chi tiết theo transmitter là bước sau, chưa cần xây framework mới.
5. **Giảm chi phí BO sau profiling.** Thử số candidate nhỏ hơn, chu kỳ fit khác hoặc warm start; kiểm tra regret/utility cùng thời gian. Không tăng budget đơn thuần để “cứu” TuRBO và bỏ qua baseline đơn giản.
6. **Fidelity thấp chỉ sau kiểm tra ranking.** Có thể sàng lọc bằng ít ray/grid thô và đánh giá finalist độ chính xác cao. Chỉ dùng khi đo được ranking ổn định; mọi phương pháp phải hưởng cùng giao thức và tính toàn bộ chi phí.

Không tự động mở rộng tilt bounds chỉ vì nghiệm chạm biên: bounds có thể là giới hạn vật lý. Chỉ mở nếu thiết bị và câu hỏi nghiên cứu cho phép. Không xây MARL chỉ để đủ hai nhánh trong README; nó hợp lý khi thật sự nghiên cứu điều khiển tuần tự, quan sát cục bộ và chi phí thay đổi góc/handover.

## 7. Lộ trình có tiêu chí hoàn thành

| Giai đoạn | Việc chính | Điều kiện chuyển bước |
|---|---|---|
| 1 — Chốt mô hình | Chọn geometric/electrical; chỉnh động cơ đa băng; xác định miền phục vụ và power budget; sửa cách báo KPI | Biến điều khiển và KPI khớp mô hình, không còn lỗi đơn vị/đếm |
| 2 — Kiểm chứng | LOS sanity check; hội tụ ray/depth; sensitivity vật liệu/noise; replay metadata | Sai số bộ giải nhỏ hơn mức cải thiện thực dụng muốn phát hiện; ranking được lượng hóa |
| 3 — Chốt câu hỏi và protocol | Chọn objective chính; baseline độc lập theo băng; định nghĩa test split và ngân sách | Protocol đóng băng trước khi chạy test; có lý do cho mọi constraint |
| 4 — Pilot rồi mở rộng | Một scene × vài seeds để ước lượng chi phí/variance; sau đó chạy ma trận chính | Đủ precision theo tiêu chí định trước; không dừng chỉ vì có p<0.05 |
| 5 — Phân tích cơ chế | Joint/per-band ablation, objective ablation, các case thất bại, lượng tử hóa góc | Giải thích được vì sao có gain và khi nào không có |
| 6 — Viết và đóng gói | Bản thảo, figures/tables sinh từ script, manifest và dữ liệu tái lập | Mọi claim nối được tới bảng hoặc thí nghiệm; người khác chạy lại được |

Nên tăng số lần lặp và độ đa dạng trước khi tăng quá sâu ngân sách một run. Có thể viết System Model và Experimental Protocol trong khi chạy pilot; Abstract/Conclusion chỉ chốt sau kết quả cuối.

## 8. Cấu trúc bản thảo và bộ bằng chứng

1. **Introduction:** bài toán vận hành cụ thể, khoảng trống so với nghiên cứu đã có, 2–3 đóng góp có thể kiểm chứng. Không liệt kê dùng Sionna/Hydra/MLflow như đóng góp khoa học.
2. **Related Work:** so các bài gần nhất theo band coupling, objective dịch vụ, realism và sample efficiency. Nêu rõ TuRBO kế thừa.
3. **System Model:** antenna, propagation, association, resource allocation, traffic, domain mask và giả định nhiễu.
4. **Problem Formulation:** biến, objective, constraints, đơn vị, lý do chọn; tính chất và phản ví dụ của J nếu vẫn nghiên cứu nó.
5. **Method:** pseudocode optimizer, input/output, cấu hình GP, điều kiện dừng và chi phí. Đánh dấu rõ phần kế thừa và phần đề xuất.
6. **Experimental Protocol:** scene, phiên bản, split, seeds, baseline, ngân sách, statistical unit, primary endpoint.
7. **Results and Discussion:** hiệu quả thực dụng, ngân sách/thời gian, ablation, robustness, failure cases; không chọn riêng run tốt nhất làm kết quả đại diện.
8. **Limitations and Conclusion:** kết luận đúng phạm vi, thực nghiệm nào còn thiếu và những điều không suy ra được.

Bộ hình/bảng gọn: bảng giả định hệ thống; bảng related work; bảng kết quả có CI; đường hội tụ theo calls/time; scatter trade-off; bản đồ trước/sau trên cùng thang màu; ablation joint/per-band; robustness theo scenario; giản đồ kiểm chứng tilt. Phụ lục chứa đầy đủ cấu hình, provenance và sensitivity dài.

Chọn tạp chí sau khi xác định đóng góp cuối: hướng ứng dụng hệ thống vô tuyến phù hợp nếu đóng góp là mô hình và đánh giá thực nghiệm; hướng thuật toán đòi hỏi cải tiến phương pháp rõ hơn. Không thể suy ra khả năng được nhận hay hạng tạp chí chỉ từ mức tăng J hiện có.

## 9. Những câu hỏi hội đồng dễ đặt ra

| Câu hỏi | Trả lời trung thực hiện nay | Bằng chứng cần bổ sung |
|---|---|---|
| TuRBO đã có, điểm mới của bạn là gì? | Cách đặt bài toán và phối hợp đa băng đang là giả thuyết đóng góp; chưa chứng minh độc lập với thuật toán | Related-work matrix và ablation coupling |
| Tại sao ba băng khác nhau phủ cùng chỗ lại xấu? | Không mặc nhiên xấu; nhiễu được mô hình hóa đồng băng, đa băng liên kết qua lựa chọn phục vụ/tài nguyên | Sửa động cơ và objective |
| Bạn tối ưu electrical hay mechanical tilt? | Mã hiện tại xoay orientation; chưa chứng minh RET tương đương | Giản đồ/power kiểm chứng hoặc đổi đúng phạm vi |
| J tăng 6.16% thì người dùng được lợi bao nhiêu? | Run này thêm 51 reports được phục vụ, failure giảm 0.506 điểm phần trăm; không có phép quy đổi J → throughput | Objective dịch vụ, offered load stress tests |
| Vì sao không chọn rule nhanh hơn và phục vụ nhiều UE hơn? | Theo dịch vụ của run hiện tại, đó là lựa chọn có lý; TuRBO chỉ thắng một số mặt | Tiêu chí quyết định chốt trước, CI và trade-off |
| Hệ thống có thể đạt J cao hơn bằng cách bỏ lớp phủ không? | Có về mặt toán học; mức khai thác bằng tilt khả thi chưa được xác định | Ablation và thí nghiệm thay góc khả thi |
| Một seed có đủ không? | Không đủ để khái quát thuật toán; hiện mới là case study | Nhiều search seeds và môi trường |
| Vì sao tin ray tracing là thực tế? | Chưa hiệu chuẩn với đo đạc; kiểm thử phần mềm không chứng minh vật lý | Sanity/convergence/calibration hoặc giới hạn claim |
| Vì sao nói 6G? | Hiện chưa có cơ sở đặc thù 6G | Bỏ nhãn 6G hoặc thêm bài toán thực sự cần nó |
| Code và kết quả có tái lập được không? | Ba bản đồ tái tính khớp KPI và 203 tests đạt; full rerun khác môi trường chưa xác minh | Scene/data hash, environment, run manifest và solver-seed validation |

Không cần tìm cách bảo vệ mọi lựa chọn hiện tại. Một lập luận có sức thuyết phục là chỉ ra rõ giả định nào được chứng minh, giới hạn nào được đo và vì sao kết luận vẫn đúng trong phạm vi đó.

## 10. Nguồn chính đã đối chiếu

- [Eriksson et al., Scalable Global Optimization via Local Bayesian Optimization, 2019](https://arxiv.org/abs/1910.01739): nguồn TuRBO.
- [Dreifuerst et al., Optimizing Coverage and Capacity in Cellular Networks using Machine Learning, ICASSP 2021](https://arxiv.org/abs/2010.13710): nghiên cứu BO/RL tối ưu tilt/power có trước.
- [Tekgul et al., Joint Uplink–Downlink Capacity and Coverage Optimization via Site-Specific Learning of Antenna Settings](https://arxiv.org/abs/2210.15732): đối chiếu formulation dịch vụ và anten.
- [BoTorch TuRBO tutorial, v0.14.0](https://botorch.org/docs/v0.14.0/tutorials/turbo_1): đối chiếu thuật toán và GP; cần ghi rõ phiên bản thực tế của dự án.
- [Sionna Radio Devices](https://nvlabs.github.io/sionna/rt/api/radio_devices.html), [Radio Map Solvers](https://nvlabs.github.io/sionna/rt/api/radio_map_solvers.html): orientation, array và precoding; đối chiếu thêm mã 2.0.1 đã cài.
- [ITU-R P.2040](https://www.itu.int/rec/R-REC-P.2040): giới hạn và nguồn mô hình vật liệu.

**Bàn giao:** chỉ thêm báo cáo thẩm định này. Các sửa đổi, ma trận thực nghiệm và hướng objective ở trên là đề xuất, chưa được hiện thực hoặc xác nhận bằng kết quả mới.
