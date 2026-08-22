"""Streamlit demo.

A UI for showing the model to people who will not call an API: enter inputs,
see a prediction, understand roughly why. Run with ``task demo``.

The demo loads the artifact through the same :class:`~app.api.inference.
ModelService` the API uses, so what it shows is what the service would return.
Do not re-implement preprocessing here.
"""

import streamlit as st


def sidebar_inputs() -> dict:
    """Render the input controls and collect their values.

    Returns:
        Feature name to value, matching the model's expected inputs.

    Raises:
        NotImplementedError: Always — implement this first.

    Notes:
        Constrain the widgets to the hard bounds declared in
        ``configs/data.yaml``. A slider that cannot produce an invalid value is
        better than an error message explaining one.
    """
    # TODO(1): one widget per feature, with min/max from the schema
    raise NotImplementedError("app.demo.streamlit_app.sidebar_inputs")


def render_prediction(features: dict) -> None:
    """Score the inputs and display the result.

    Args:
        features: Values collected from the sidebar.

    Raises:
        NotImplementedError: Always — implement this first.

    Notes:
        Cache the model service with ``@st.cache_resource`` — Streamlit re-runs
        the whole script on every interaction, and reloading the artifact each
        time makes the demo feel broken.
    """
    # TODO(1): service = get_cached_service()
    # TODO(2): prediction = service.predict(to_frame([features]))[0]
    # TODO(3): st.metric(...) plus a short explanation of the inputs' effect
    raise NotImplementedError("app.demo.streamlit_app.render_prediction")


def main() -> None:
    """Lay out the page.

    Raises:
        NotImplementedError: Always — implement this first.
    """
    st.set_page_config(page_title="<project-name> demo", layout="wide")
    st.title("<project-name>")
    st.caption("Demo interface — predictions come from the same artifact the API serves.")
    # TODO(1): features = sidebar_inputs()
    # TODO(2): render_prediction(features) behind a button
    raise NotImplementedError("app.demo.streamlit_app.main")


if __name__ == "__main__":
    main()
