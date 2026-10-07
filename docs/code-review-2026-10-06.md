# Review code và mô hình — 06/10/2026

Phạm vi: checkout `80c4ef1`, README, report, ADR, 52 module Python trong `src`, demo, cấu hình, task/DVC, tests và các đoạn điều phối trong notebook. Không sửa implementation. P1: ảnh hưởng bản chất bài toán hoặc độ tin cậy kết quả; P2: lỗi có điều kiện hoặc sai lệch mô hình/tái lập; P3: bảo trì và diễn đạt.

Pipeline hiện tại sinh scenario → ray-trace từng band → kiểm tra/ép kiểu dữ liệu → tìm absolute tilt bằng Sobol hoặc TuRBO → công bố shortlist → so sánh artifacts. J chỉ đọc RSRP theo tile, không đọc demand hay throughput. Vì vậy tính đúng của KPI, tính đúng của objective so với công thức, và tính phù hợp của objective với mục tiêu nghiên cứu là ba câu hỏi khác nhau.

## Các phát hiện cần xử lý

### 1. [P1] Biến được tối ưu là mechanical rotation, nhưng báo cáo diễn giải thành electrical tilt/RET

Vị trí: [transmitter.py:34](D:/Projects/band-tilt/src/simulation/transmitter.py:34), [radio.py:377](D:/Projects/band-tilt/src/simulation/radio.py:377), [report.md:84](D:/Projects/band-tilt/docs/report.md:84).

`tilt_deg` được gán vào pitch của `Transmitter.orientation`. Toàn bộ hệ tọa độ anten, pattern phần tử và hình học array quay theo nó. `RadioMapSolver` không nhận `precoding_vec` theo tilt; mặc định là các trọng số đồng pha đã chuẩn hóa. Đây không phải cách hiện thực electrical steering mà báo cáo đang gọi là đầu vào cho RET.

TR 38.901 §7.1.4 mô tả mechanical downtilt bằng quay hệ tọa độ; §7.3.1 dùng trọng số phức cho electrical steering. [3GPP TR 38.901 V19.1.0](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf). Mặc định precoding cũng được xác nhận trong [source chính thức của Sionna](https://github.com/NVlabs/sionna-rt/blob/main/src/sionna/rt/radio_map_solvers/radio_map_solver.py) và bản đã cài tại workspace.

**Tác động:** optimizer có thể tối ưu đúng hàm đang chạy nhưng giải một bài toán vật lý khác. Hai kiểu tilt không tương đương ở ngoài boresight, sidelobe và phân cực. Kết luận về RET chưa được implementation hỗ trợ.

**Hướng sửa:** nếu yêu cầu là RET, giữ orientation cơ khí cố định và truyền precoding theo electrical tilt cho từng transmitter/band. Thêm một check pattern đơn giản: góc búp chính đổi đúng dấu/độ lớn, tổng công suất giữ nguyên. Nếu chỉ nghiên cứu quay panel, đổi mô tả và phạm vi kết luận; không chỉ đổi tên biến rồi vẫn gọi kết quả là electrical tilt.

### 2. [P1] Evaluation trộn KPI lịch sử với config và dữ liệu hiện tại

Vị trí: [evaluation/run.py:98](D:/Projects/band-tilt/src/evaluation/run.py:98), [evaluation/run.py:122](D:/Projects/band-tilt/src/evaluation/run.py:122), [compare.py:366](D:/Projects/band-tilt/src/evaluation/compare.py:366), [capacity.py:88](D:/Projects/band-tilt/src/kpi/capacity.py:88).

Scoreboard lấy `best_kpi`/`incumbent_kpi` từ `run.json`, nhưng bản đồ, per-band KPI, serving và sector-impact được tái tính bằng `cfg`, UE parquet và sectors CSV đang có trên đĩa. `verify(runs, baseline)` không nhận config/UE/sector hiện tại để kiểm tra chúng. `CapacitySpec` chỉ đối chiếu số lượng sector, không đối chiếu tên và thứ tự transmitter.

**Trigger:** đổi `kpi.hole_dbm`, PRB share, UE parquet hoặc thứ tự sector rồi chạy lại `task evaluate`. Một báo cáo có thể chứa scoreboard theo định nghĩa cũ và bản đồ/throughput theo định nghĩa mới. Đảo thứ tự sector cùng số lượng có thể gán PRB sai transmitter. Bảng `kpi_reproducibility` chỉ ghi gap, không chặn xuất báo cáo.

**Hướng sửa:** tái tính từ snapshot input của run, hoặc bắt buộc current inputs khớp fingerprint và config có ảnh hưởng đến kết quả. Căn sector bằng `tx_name`, không bằng vị trí dòng. Nếu cho phép đánh giá lại theo định nghĩa mới, ghi rõ đây là phép đánh giá mới và tính lại cả scoreboard. Không trộn hai cách.

### 3. [P1] Comparability và scenario ID chưa chứng minh các run đo cùng một thế giới

Vị trí: [runs.py:23](D:/Projects/band-tilt/src/evaluation/runs.py:23), [runs.py:207](D:/Projects/band-tilt/src/evaluation/runs.py:207), [radio.py:265](D:/Projects/band-tilt/src/simulation/radio.py:265), [scenario/run.py:36](D:/Projects/band-tilt/src/scenario/run.py:36).

`verify` bỏ qua `solver_seed`, `tx_name`, vị trí/azimuth sector, cấu hình TX array, incumbent tilt, bounds và nội dung scene/UE. `_kpi_definition` so noise/capacity giữa các run, nhưng baseline archive không lưu đủ các trường để so với baseline. Không có version riêng cho objective; report đã thừa nhận phần này. `scenario_id` hash config và đường dẫn scene, không hash scene XML/mesh hay các file đầu vào thực tế. Sửa geometry cùng đường dẫn vẫn giữ ID.

**Đã tái hiện:** đổi tên transmitter và solver seed trong run nhưng giữ các metadata đang được so sánh: tất cả comparability checks vẫn `True`.

**Tác động:** sai lệch do solver noise, thay anten, thay world hoặc thay definition có thể bị diễn giải thành gain do tilt. `seed_summary` còn dùng incumbent của run đầu cho mọi run, dù tính đồng nhất đó chưa được kiểm chứng.

**Hướng sửa:** lưu fingerprint của scene cùng assets, sectors và UE; lưu TX configuration, noise/SCS/bandwidth, objective/schema version, solver seed, baseline tilt/bounds. Phân biệt seed tìm kiếm với seed mô phỏng. Dùng các trường này để nhóm experiment trước khi chọn run mới nhất. Nếu chủ đích dùng nhiều solver seed, so delta với incumbent tương ứng của từng run.

### 4. [P1] Optimizer bỏ qua các kiểm tra đầu vào mà simulation yêu cầu

Vị trí: [evaluator.py:123](D:/Projects/band-tilt/src/optim/evaluator.py:123), [radio.py:221](D:/Projects/band-tilt/src/simulation/radio.py:221), [radio.py:427](D:/Projects/band-tilt/src/simulation/radio.py:427).

`Evaluator` đọc parquet rồi dùng ngay, không so `scenario_id` trong UE với manifest. `_check_tilt_table` kiểm tra PRB theo bandwidth/SCS chỉ được gọi ở `radio.solve`; optimizer gọi `solve_bands` trực tiếp và bỏ qua nó. Vì các stage được chạy độc lập, việc preprocess từng thành công không bảo đảm file hiện tại còn đúng.

**Đã tái hiện:** dùng bản sao UE có `scenario_id='WRONG_SCENARIO'` và bản sao sector có `max_prb=1` cho mọi band. Khi mock riêng scene loader để không chạy GPU, `Evaluator` khởi tạo thành công. Validator của simulation từ chối chính sector table đó với yêu cầu 216/106/52 PRB.

**Hướng sửa:** kiểm tra contract một lần lúc tạo evaluator, trước khi load GPU scene. Dùng chung validation cho setup của simulation và optimization; so scenario ID/fingerprint của UE. Không cần chạy lại toàn bộ preprocessing hay validation ở mỗi candidate.

### 5. [P2] Schema checker chấp nhận map sai và có thể crash thay vì báo lỗi contract

Vị trí: [schema.py:132](D:/Projects/band-tilt/src/data/schema.py:132), [schema.py:175](D:/Projects/band-tilt/src/data/schema.py:175), [schema.py:225](D:/Projects/band-tilt/src/data/schema.py:225), [schema.py:275](D:/Projects/band-tilt/src/data/schema.py:275).

Kiểm tra shape chỉ là `sinr.shape == rsrp.shape`, không phải shape mong đợi `(n_band, n_tx, n_rows, n_cols)`. Mask NaN khớp nhau không chứng minh các giá trị còn lại finite. `_numeric` chỉ thử chuyển kiểu, không kiểm tra schedule không rỗng, số hữu hạn và kích thước dương.

**Đã tái hiện trên fixtures:**

- Manifest 2×2 nhưng RSRP và SINR cùng shape `(1,1,1,1)`: mọi check đạt.
- SINR toàn `+inf` trong khi RSRP finite: mọi check đạt; serving sau đó bỏ UE có tín hiệu vì rate không finite.
- Schedule `[]`: `verify` phát sinh `IndexError`, trái contract hứa trả bảng kiểm tra.

**Hướng sửa:** kiểm tra expected shape, dtype và finite-or-NaN trước các phép reduction; chỉ cho NaN theo quy ước no-path. Kiểm tra grid/schedule hợp lệ trước indexing. Giữ lỗi cấu trúc thành check thất bại rồi dừng các check phụ thuộc.

### 6. [P2] Cấu hình anten thu dipole không tác động đến kết quả radio map

Vị trí: [radio.py:477](D:/Projects/band-tilt/src/simulation/radio.py:477), [simulation.yaml:82](D:/Projects/band-tilt/configs/simulation.yaml:82).

Code dựng `scene.rx_array` từ config, nhưng `RadioMapSolver` tính bản đồ với bộ thu isotropic lý tưởng; đường solve này không áp pattern dipole của RX. Đã kiểm tra source thư viện cài trong workspace và [source Sionna chính thức](https://github.com/NVlabs/sionna-rt/blob/main/src/sionna/rt/radio_map_solvers/radio_map_solver.py).

**Tác động:** người dùng đổi receiver pattern/polarization và tưởng mình đã thay mô hình UE nhưng KPI không thay theo cấu hình đó. Đây vừa là config không có hiệu lực, vừa là sai lệch mô tả mô hình.

**Hướng sửa:** ghi rõ radio map dùng isotropic RX và bỏ/đánh dấu phần receiver không có hiệu lực trong workflow này. Chỉ bổ sung receiver-aware path evaluation nếu bài toán thực sự yêu cầu pattern UE; không giả định set `rx_array` đã đủ.

### 7. [P2] Notebook cache kết quả bằng method/seed nên có thể bỏ qua experiment mới

Vị trí: [03a_baseline.ipynb:676](D:/Projects/band-tilt/notebooks/03a_baseline.ipynb:676), [03b_turbo.ipynb:737](D:/Projects/band-tilt/notebooks/03b_turbo.ipynb:737), [runs.py:176](D:/Projects/band-tilt/src/evaluation/runs.py:176).

03a không lọc scenario trước khi tạo `done`; 03b có lọc scenario nhưng vẫn không xét budget, fidelity hoặc objective version. Thay budget từ 144 lên 288 ở cùng seed có thể chỉ đọc lại run cũ. Đổi scenario khiến 03a bỏ chạy rồi fail comparability. CLI evaluation cũng chọn newest theo `(method, seed)` trước khi lọc experiment, nên một experiment mới khác có thể che run tương thích cũ.

**Hướng sửa:** dùng một hàm chọn/cache run chung theo experiment fingerprint và method/seed, kèm budget/search settings khi quyết định tái sử dụng. Có cờ chạy lại rõ ràng. Không bắt người dùng xóa toàn bộ lịch sử để chạy một experiment mới.

### 8. [P2] DVC không theo dõi đầy đủ cấu hình ảnh hưởng optimization

Vị trí: [dvc.yaml:111](D:/Projects/band-tilt/dvc.yaml:111), [optim/base.yaml:17](D:/Projects/band-tilt/configs/optim/base.yaml:17).

Budget thực nằm ở `configs/optim/base.yaml`, nhưng DVC chỉ theo dõi `seed` và `n_solutions` của file này. Method file chứa chuỗi interpolation `${optim.budget...}`. Ngoài ra thiếu `candidates` của TuRBO và lựa chọn method trong `configs/config.yaml.defaults`. Khi các mục này đổi, dependency graph chưa phản ánh đủ experiment mới.

**Tác động:** sau khi DVC được khởi tạo, cơ chế skip unchanged có thể giữ kết quả cũ khi budget/pool/method thay đổi. Repo hiện chưa khởi tạo DVC nên đây là lỗi workflow có điều kiện, không phải lỗi đã xảy ra trong run được báo cáo.

**Hướng sửa:** thêm các params thiếu, hoặc theo dõi nguyên file config nhỏ thay vì duy trì whitelist dễ sót. DVC cũng cần đường output ổn định hoặc manifest trỏ run hoàn tất nếu muốn kiểm soát artifact timestamped.

### 9. [P2] Objective có bước nhảy tại ngưỡng rival, trái mô tả “no step”

Vị trí: [overlap.py:87](D:/Projects/band-tilt/src/kpi/overlap.py:87), [report.md:192](D:/Projects/band-tilt/docs/report.md:192).

Rival chỉ đóng góp khi `R_i > hole_dbm`. Ngay khi đi qua ngưỡng, một lượng công suất khác 0 được thêm vào mẫu số. Strength factor của strongest không triệt được bước nhảy này.

**Đã tái hiện:** một band, strongest −119 dBm; rival đổi từ −120.000001 thành −119.999999 dBm. Tile score giảm từ **0.0333333 xuống 0.0185771**, dù thay đổi tín hiệu chỉ 0.000002 dB và tile vẫn covered.

**Tác động:** phát biểu utility không có step không đúng; objective nhạy quanh ngưỡng và GP có thể phải fit một bước nhảy. Điều này độc lập với tính non-monotone giữa band đã được ADR thừa nhận.

**Hướng sửa:** ít nhất sửa tài liệu và thêm boundary check. Nếu cần utility liên tục, tính công suất của mọi đường finite hoặc taper trọng số rival về 0 tại ngưỡng; đây là thay đổi định nghĩa J, phải version và chạy lại comparisons.

### 10. [P2] Tham số hotspot mass không bằng tỷ lệ hotspot trung bình như mô tả

Vị trí: [traffic.py:133](D:/Projects/band-tilt/src/scenario/traffic.py:133), [traffic.py:170](D:/Projects/band-tilt/src/scenario/traffic.py:170).

Với tổng intensity hotspot là H và background cố định B, tỷ lệ thực là `H/(B+H)`. Đặt `E[H]` bằng số hotspot không làm `E[H/(B+H)]` bằng `E[H]/(B+E[H])`. Hàm này lõm khi B>0, nên dao động làm trung bình giảm.

**Đã tái hiện:** giữ amplitude 0.6, rho 0.85, sigma 0.25, 4 hotspots, target 0.7; sinh schedule 365 ngày, seed 42 cho trung bình mass **0.6666615**. Đây là mass của schedule, không phải sai số đếm vài UE.

**Hướng sửa:** nếu 0.7 chỉ là tỷ lệ khi intensity ở mức tham chiếu, đổi tên/mô tả cho đúng. Nếu là mean thực của schedule, giải một scalar B sao cho trung bình `H/(B+H)` bằng target, xử lý riêng target 0 và 1. Không cần thay toàn bộ mô hình traffic.

### 11. [P2] Thống kê chi phí bỏ một solve và quy overhead sai cho GP

Vị trí: [optim/run.py:69](D:/Projects/band-tilt/src/optim/run.py:69), [history.py:285](D:/Projects/band-tilt/src/optim/history.py:285), [report.md:522](D:/Projects/band-tilt/docs/report.md:522).

Khi lưu best map, run ray-trace winner thêm một lần. `n_evaluations` và `ray_tracing_seconds` chỉ tính history, còn wall-clock có cả lần trace này, load scene, tính KPI, phục vụ UE và ghi map. Vì vậy `wall − ray_tracing` không phải riêng GP fitting/acquisition. Random search không có GP nhưng vẫn có chênh lệch này.

**Hướng sửa:** giữ 145 là search evaluations, ghi thêm 1 archive evaluation và thời gian của nó. Tách `search`, `scoring`, `proposal/model`, `archive` nếu muốn kết luận về overhead; nếu chưa đo riêng, chỉ gọi phần chênh là non-search-ray-tracing overhead.

### 12. [P2] Report có kết luận coverage sai và liên kết artifact không bất biến

Vị trí: [report.md:391](D:/Projects/band-tilt/docs/report.md:391), [report.md:510](D:/Projects/band-tilt/docs/report.md:510), [report.md:618](D:/Projects/band-tilt/docs/report.md:618), [evaluation/run.py:34](D:/Projects/band-tilt/src/evaluation/run.py:34).

Câu “Both methods lowered the hole rate” mâu thuẫn Table 4: random tăng từ khoảng 0.1125 lên 0.1134. Phần bên dưới còn ghi random mở 463 hole và đóng 374 hole. Đây là lỗi biên tập có thể xác nhận mà không chạy simulator.

Báo cáo ghi run ngày 02/10, nhưng nguồn `method_cost.csv` hiện trỏ run ngày 05/10. Table 11 ghi wall-clock random/TuRBO 12.94/11.10 phút; CSV hiện là **9.03/11.43 phút**. Outputs run 02/10 nêu trong report không có ở workspace hiện tại. Điều này không chứng minh số lịch sử sai; nó chứng minh link nguồn hiện không còn xác minh số lịch sử đó.

**Hướng sửa:** sửa câu coverage; xuất report tables/figures vào thư mục theo experiment/run-set và trỏ report vào snapshot đó. Gắn danh sách run ID cùng config/version trong manifest của báo cáo. Chỉ dùng thư mục `latest` cho giao diện xem mới nhất.

### 13. [P2] Ép tile index sang int16 có thể làm hỏng dữ liệu đã qua validation

Vị trí: [build.py:25](D:/Projects/band-tilt/src/data/build.py:25), [build.py:49](D:/Projects/band-tilt/src/data/build.py:49).

Schema không giới hạn kích thước grid theo int16 nhưng builder ép `tile_row`/`tile_col` sang int16. Giá trị **40000 thành −25536** trong probe. Grid hiện tại 326×310 không bị ảnh hưởng; grid độ phân giải cao hoặc dữ liệu thật có thể bị.

**Hướng sửa:** dùng int32 cho tile index, hoặc kiểm tra range trước cast và báo lỗi rõ ràng. Số byte tiết kiệm ở bảng UE nhỏ không đáng đổi lấy silent overflow. Cũng cần xét range của optional `component` nếu giữ int16.

## Clean code, phụ thuộc, redundant và deprecated

Không phát hiện vòng import trực tiếp trong graph 52 module, kể cả các re-export được phân tích. Không có cơ sở gọi repository đang lỗi circular import. Tuy nhiên có các điểm coupling đáng sửa:

| Mức | Vị trí | Vấn đề và hướng xử lý |
|---|---|---|
| P2 | [compare.py:17](D:/Projects/band-tilt/src/evaluation/compare.py:17), [methods/base.py:19](D:/Projects/band-tilt/src/optim/methods/base.py:19), [history.py:22](D:/Projects/band-tilt/src/optim/history.py:22) | Evaluation import một constant/type nhưng kéo theo concrete evaluator và simulation. Runtime probe xác nhận `src.optim.evaluator` đã có trong `sys.modules`, trái docstring evaluation. Sionna vẫn lazy import nên chưa bắt GPU. Đưa import chỉ dùng type vào `TYPE_CHECKING`; đặt vocabulary history ở nơi không cần runtime dependency lên evaluator. Không cần tạo hệ thống interface mới. |
| P2 | [capacity.py:64](D:/Projects/band-tilt/src/kpi/capacity.py:64) | Domain KPI tự đọc CSV/config; evaluator đọc sectors trong TiltSpace rồi CapacitySpec đọc lại. Truyền sectors/spec đã load và căn tên transmitter sẽ vừa giảm I/O, vừa sửa nguyên nhân gán nhầm PRB. Đọc một snapshot đầu run, không để các lần đọc thấy trạng thái file khác nhau. |
| P3 | [compare.py:26](D:/Projects/band-tilt/src/evaluation/compare.py:26), [plotting.py:7](D:/Projects/band-tilt/src/utils/plotting.py:7) | Tính bảng phụ thuộc module plotting chỉ để lấy label, kéo theo matplotlib. Tách mapping label khỏi pyplot hoặc áp label ở export/presentation; giữ computation trả tên machine-readable. |
| P3 | [methods/__init__.py:21](D:/Projects/band-tilt/src/optim/methods/__init__.py:21) và thư mục random/turbo | Mỗi method có một file implementation cộng package re-export. Có thể làm phẳng thành `methods/random.py`, `methods/turbo.py`; đây là lựa chọn bảo trì, không phải bug. Protocol cho evaluator có giá trị thật vì tests dùng stub, không nên xóa chỉ vì implementation production có một cái. |
| P3 | [materials.py:80](D:/Projects/band-tilt/src/simulation/materials.py:80) | Phụ thuộc private ITU table của Sionna dễ vỡ khi upgrade. Lockfile giảm rủi ro, và code đã báo lỗi import rõ. Nên có smoke check cho bảng/hệ số đang dùng khi nâng thư viện. Chưa có bằng chứng API này đã deprecated. |

Comments/docs đang lỗi thời hoặc gây hiểu nhầm:

- [evaluator.py:77](D:/Projects/band-tilt/src/optim/evaluator.py:77) nói constructor kiểm tra mast trên open ground; nó chỉ load scene/sector/UE, không có ground check. Ground check nằm ở generator và không bảo vệ sector data thay bằng dữ liệu thật.
- [radio.py:29](D:/Projects/band-tilt/src/simulation/radio.py:29) nói no-path NaN bị loại khỏi mọi reduction; thực tế hole/J vẫn tính những tile này như không phủ. Chỉ các quality percentile loại chúng.
- [capacity.py:195](D:/Projects/band-tilt/src/kpi/capacity.py:195) mô tả các đơn giản hóa đều optimistic và giữ vì search cần smooth throughput. J không dùng throughput; bỏ spatial multiplexing cũng không mặc nhiên optimistic. Sửa lý do và hướng bias theo từng giả định.
- [test_kpi.py:257](D:/Projects/band-tilt/tests/test_kpi.py:257) tên test nói mất layer không bao giờ tăng score, nhưng docstring và objective cho phép điều đó với layer yếu. Đổi tên đúng case và giữ thêm counterexample của layer yếu để tránh đọc nhầm guarantee.
- [README.md:230](D:/Projects/band-tilt/README.md:230) ghi 261 tests, hiện chạy được 276; phần testing nói chạy pytest ở pre-commit nhưng hook file chỉ có các check file và Ruff. README còn mô tả utils có seeding, trong khi seeding ở simulation.
- [pyproject.toml:15](D:/Projects/band-tilt/pyproject.toml:15) còn văn bản “every new project generated from this template”. [pre-commit config](D:/Projects/band-tilt/.pre-commit-config.yaml:15) pin Ruff 0.14.0 trong khi dev yêu cầu >=0.16: hai lối kiểm tra dùng tool khác phiên bản. Chưa quan sát khác output trên code hiện tại, nhưng nên đồng bộ.
- ADR 0001 còn weighted score, tolerance và “No Pareto front”. Nó có chỉ rõ bị supersede một phần, nên đây là lịch sử hợp lệ, không phải dead doc cần xóa. Nên đánh dấu rõ đoạn nào hết hiệu lực, đặc biệt Pareto dùng cho chẩn đoán hiện có trong compare.
- Các wrapper KPI nhỏ và phương thức dùng từ notebook không tự động là dead code. Không thấy bằng chứng đủ chắc để đề nghị xóa hàng loạt.

Hai `DeprecationWarning` khi chạy tests đến từ `linear_operator.utils.linear_cg` gọi `torch.jit.script`; không phải lời gọi deprecated trong source dự án. Theo dõi tương thích bộ torch/linear_operator/GPyTorch khi nâng dependency, không thay code search theo tên warning một cách máy móc.

## Đối chiếu 3GPP và giới hạn nghiên cứu

1. **Các giá trị PRB mặc định đúng:** 10/20/40 MHz ở 15 kHz tương ứng 52/106/216 theo Table 5.3.2-1. Nhưng `_N_RB` không ghi release và thiếu 3/7 MHz có trong bản V19.3.1; probe 3 MHz bị từ chối. Ghi phạm vi hỗ trợ/release, không cần mở rộng nếu chưa dùng. N_RB hợp lệ chưa kiểm chứng operating band hay channel raster. [TS 38.101-1 V19.3.1](https://www.etsi.org/deliver/etsi_ts/138100_138199/13810101/19.03.01_60/ts_13810101v190301p.pdf).

2. **12 subcarriers/PRB đúng.** `12 × SCS` là bandwidth dùng trong Shannon proxy, không phải toàn bộ công thức data-rate NR. [TS 38.211 V19.1.0 §4.4.4.1](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/19.01.00_60/ts_138211v190100p.pdf). Công thức TS 38.306 còn có layers, modulation/coding, symbol rate và overhead. Không coi Shannon proxy là bug vì README/ADR đã công bố đơn giản hóa này. [TS 38.306 V17.0.0 §4.1.2](https://www.etsi.org/deliver/etsi_ts/138300_138399/138306/17.00.00_60/ts_138306v170000p.pdf).

3. **RSRP/SINR đang là proxy vật lý với giả định flat per-RE.** SS-RSRP/SS-SINR trong chuẩn gắn với các RE/reference signal và điều kiện đo cụ thể. Chỉ đặt công suất theo RE và noise kT·SCS chưa hiện thực đầy đủ phép đo SS. Nên ghi rõ proxy và giả định; không đổi noise sang cả channel bandwidth khi signal vẫn per-RE. [TS 38.215 V19.2.0 §5.1.1, §5.1.5](https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/19.02.00_60/ts_138215v190200p.pdf).

4. **Không nên coi overlap khác band đương nhiên là nhiễu.** Khuyến nghị inter-band interference cần nêu cơ chế RF cụ thể, chẳng hạn leakage, blocking, intermodulation. Chuẩn có yêu cầu receiver riêng cho các hiệu ứng đó; không cộng toàn bộ công suất 700/1800/2600 MHz vào cùng mẫu số SINR. Đây là khuyến nghị về phạm vi mô hình, không phải bằng chứng code thiếu nhiễu bắt buộc. [TS 38.101-1 §7.5–7.8](https://www.etsi.org/deliver/etsi_ts/138100_138199/13810101/19.03.01_60/ts_13810101v190301p.pdf).

5. **Non-monotonicity và không tối ưu demand là quyết định đã công khai.** Implementation J khớp công thức ADR, nhưng tăng J không bảo đảm giảm hole/overlap hay giữ vai trò từng band. Nếu đó là điều kiện bắt buộc của bài toán, cần constraint/acceptance rule ngoài J. Nếu là mục tiêu kỳ vọng, dùng cách diễn đạt “proxy objective; kiểm tra trade-off bằng KPI”. Không tự ý đổi objective trong một lần cleanup.

6. **Một seed, một scenario, chưa held-out** đã được tài liệu công bố. Không ghi nhận lại thành bug implementation mới. Thống kê nhiều seed/held-out cần thiết để mở rộng kết luận, nhưng không thay thế việc sửa sai lệch tilt vật lý và provenance trước.

## Kiểm chứng đã thực hiện

- `.venv/Scripts/python.exe -m ruff check .`: đạt.
- `.venv/Scripts/python.exe -m ruff format --check .`: 71 files đã đúng format.
- Toàn bộ pytest: **276 passed, 2 dependency deprecation warnings**, 56.19 giây ở lượt cuối.
- Lượt pytest mặc định bị lỗi quyền thư mục tạm sandbox; lượt chỉ đổi `--basetemp` vẫn vướng tempfile của MLflow. Lượt cuối đặt cả `tempfile.tempdir` trong workspace và dùng `--basetemp` riêng, toàn bộ đạt. Không ghi nhận các lỗi môi trường này là bug dự án.
- Đã thực hiện probes synthetic cho shape/SINR/schedule, comparability, int16 overflow, discontinuity, traffic mean; dùng bản sao inputs thật và mock scene loader cho validator evaluator.
- AST graph: không thấy explicit import cycle trong 52 module.
- `task --dry pipeline -- seed=7`: override thực sự được chuyển đến mọi stage; không có lỗi forwarding CLI như có thể nghi ngờ khi chỉ nhìn YAML.
- Chưa chạy lại GPU ray tracing, full optimization hoặc Colab. Review tilt/RX dựa trên luồng code và source/định nghĩa nhà cung cấp, không giả vờ đã đo một experiment vật lý mới.
- Chưa tái lập số lịch sử ngày 02/10 vì workspace hiện chỉ có run ngày 05/10. Đã kiểm tra metadata và các CSV hiện có, không sửa chúng.

Thứ tự sửa đề xuất: **(1) xác định đúng electrical/mechanical tilt → (2) input validation và provenance/comparability → (3) snapshot báo cáo và cache experiment → (4) objective/traffic boundary cases → (5) cleanup import, config và comments.** Sau các thay đổi làm đổi mô hình/J, phải tạo experiment version mới và chạy lại kết quả.
