from inspect import signature


def patch_lightgbm_sklearn_validation():
    """Adapt LightGBM <4.6 validation aliases to scikit-learn 1.8+."""
    try:
        import lightgbm.sklearn as lightgbm_sklearn
    except ImportError:
        return

    for name in ("_LGBMCheckXY", "_LGBMCheckArray"):
        validation = getattr(lightgbm_sklearn, name, None)
        if validation is None:
            continue
        try:
            parameters = signature(validation).parameters
        except (TypeError, ValueError):
            continue
        if "force_all_finite" in parameters or "ensure_all_finite" not in parameters:
            continue

        def validation_compat(*args, _validation=validation, force_all_finite=True, **kwargs):
            kwargs.setdefault("ensure_all_finite", force_all_finite)
            return _validation(*args, **kwargs)

        setattr(lightgbm_sklearn, name, validation_compat)
