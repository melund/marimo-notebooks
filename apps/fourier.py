import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _(all_df, np, pl, ui, ui_dropdown):
    raw_data = all_df.select(pl.col(ui_dropdown.value))


    def offset_correction(y):
        return y - y.mean()


    def linear_correction(y):
        """Add linear correction to y so that the last element coincides with the first."""
        if len(y) < 2:
            return y
        correction = np.linspace(0.5*(y[-1]-y[0]), 0.5*(y[0]-y[-1]), len(y))
        return y + correction

    normalized_data = raw_data

    if ui["checkbox_offset"].value:   
        for label in raw_data.columns:
            normalized_data = normalized_data.with_columns(
                pl.col(label).map_batches(offset_correction, return_dtype=pl.Float64 ).alias(label)
            )
    if ui["checkbox_linear"].value:   
        for label in raw_data.columns:
            normalized_data = normalized_data.with_columns(
                pl.col(label).map_batches(linear_correction, return_dtype=pl.Float64 ).alias(label)
            )

    # normalized_data = raw_data
    # normalized_data
    return normalized_data, raw_data


@app.cell
def _(chart, mo, ui, ui_dropdown, ui_file):
    mo.vstack([ui_file, ui_dropdown, ui.hstack(justify= "start"), chart])

    return


@app.cell
def _(file, np, pl, ui_file):
    # if ui_file.value:
    file_name = file.value
    # else:
    #     file_name = "apps/2014001_C2_03.npz"

    with np.load(file_name) as npz_file:
        _alldata = dict(npz_file.items())

    _time_var = "Main.Studies.InverseDynamicStudy.Output.Abscissa.t"

    _dof_val = "Main.Studies.InverseDynamicStudy.Output.ModelEnvironmentConnection.JointsAndDrivers.AllDoFs.Values"
    _dof_var = "Main.Studies.InverseDynamicStudy.ModelEnvironmentConnection.JointsAndDrivers.AllDoFs.HeaderNames"
    _extra_val = "Main.Studies.InverseDynamicStudy.Output.ExtraOutput"
    _extra_var = "Main.Studies.InverseDynamicStudy.ExtraOutputLabel"
    _fs1 = _alldata.pop("Main.Studies.InverseDynamicStudy.GaitCycle.Right.FootStrike1")
    _fs2 = _alldata.pop("Main.Studies.InverseDynamicStudy.GaitCycle.Right.FootStrike2")

    _time = _alldata[_time_var]
    _trim_wnd =  slice(np.argmin(np.abs(_time - _fs1)),  np.argmin(np.abs(_time - _fs2)))

    all_df = pl.concat([
        pl.from_numpy(
            data = _alldata[_dof_val][_trim_wnd],
            schema = _alldata[_dof_var].tolist(),
        ),
        pl.from_numpy(
            data = _alldata[_extra_val][_trim_wnd],
            schema = _alldata[_extra_var].tolist(),
        )],
        how="horizontal"
    )

    _included_const_columns = ["Anthropometrics", "MetaInfo"]
    for col in _alldata:
        if all(name not in col for name in _included_const_columns) or np.ndim(_alldata[col]) != 0:
            continue
        new_name = col.removeprefix("Main.ModelSetup.MetaInfo.")
        all_df = all_df.with_columns(
            pl.lit(_alldata[col]).first().alias(new_name)
        )

    return (all_df,)


@app.cell
def _(alt, mo, normalized_data, np, pl, raw_data, sillywalk, ui):
    _n_modes = ui['silder_modes'].value

    fourier_coefficients = sillywalk.anybody.compute_fourier_coefficients(
        normalized_data, _n_modes
    )
    A = fourier_coefficients.to_numpy()[0, :_n_modes]
    B = np.insert(
        fourier_coefficients.to_numpy()[0, _n_modes:],
        0,
        0,
    )

    def reconstruct_signal(A, B, length):
        """Reconstruct the signal from A/B coefficients
        similar to how AnyBody' CosSin fourier driver
        """
        t = np.linspace(0, 1, length, endpoint=False)  # np.arange(length)/length
        harmonics = np.arange(len(A))[:, None] * 2 * np.pi * t
        return A @ np.cos(harmonics) + B @ np.sin(harmonics)

    nsteps = len(normalized_data)
    reconstructed = reconstruct_signal(
        A, B, nsteps
    )  # ((A @ np.cos(harmonics) + B @ np.sin(harmonics))

    source = pl.DataFrame(
        {
            "steps": np.arange(nsteps),
            "original": raw_data.to_numpy()[:,0],
            "normalized": normalized_data.to_numpy()[:, 0],
            "reconstructed": reconstructed,
        
        }
    ).unpivot(index="steps", variable_name="Signal")
    # source

    chart = mo.ui.altair_chart(
        alt.Chart(source).mark_line().encode(x="steps", y="value", color="Signal").interactive()
    )

    return (chart,)


@app.cell
def _(all_df, mo):

    ui_dropdown = mo.ui.dropdown(options=all_df.columns, value=all_df.columns[0], label="Variable:")

    ui = mo.ui.dictionary(
      {"silder_modes": mo.ui.number(1, 25, value = 4, label="modes"),
       "checkbox_offset": mo.ui.checkbox(label = "Offset correction"),
       "checkbox_linear": mo.ui.checkbox(label="linear correction"),
      }
    )

    return ui, ui_dropdown


@app.cell
def _(mo):
    ui_file = mo.ui.file(label="Upload trial data")

    return (ui_file,)


@app.cell
def _():
    import altair as alt
    import numpy as np
    import polars as pl
    import sillywalk

    return alt, np, pl, sillywalk


@app.cell
def _():
    import marimo as mo


    return (mo,)


if __name__ == "__main__":
    app.run()
