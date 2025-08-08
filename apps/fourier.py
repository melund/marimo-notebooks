# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo==0.14.16",
#     "altair==5.5.0",
#     "pandas==2.3.1",
#     "numpy==2.3.2",
#     "requests==2.32.4",
#     "sillywalk==1.0.0a1",
#     "scipy==1.16.0"
# ]
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    ui_file = mo.ui.file(label="Upload trial data")
    ui_file
    return (ui_file,)


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

    def windowed_correction(y):
        """Add Tukey window correction to y so that the last element coincides with the first."""
        N = len(y)
        if N < 2:
            return y
        wl = int(N*ui["number_window"].value)
        hp_wnd = hann_poisson_window(wl*2, 2)
        window = np.concatenate((hp_wnd[-wl:], np.zeros(N-2*wl), hp_wnd[:wl]))
        lin_correction = np.linspace(0.5*(y[-1]-y[0]), 0.5*(y[0]-y[-1]), N)
        correction = np.multiply(lin_correction, window)
        return y + correction

    def hann_poisson_window(N, alpha):
        n_values = np.arange(1, N + 1)
        # Calculate the Hann part
        hann_part = 0.5 * (1 - np.cos(2 * np.pi * n_values / N))
        # Calculate the Poisson part
        poisson_part = np.exp(-alpha * np.abs(N - 2 * n_values) / N)
        return hann_part * poisson_part


    corrected_data = raw_data

    if ui["checkbox_offset"].value:   
        for label in raw_data.columns:
            corrected_data = corrected_data.with_columns(
                pl.col(label).map_batches(offset_correction, return_dtype=pl.Float64 ).alias(label)
            )
    if ui["checkbox_linear"].value:   
        for label in raw_data.columns:
            corrected_data = corrected_data.with_columns(
                pl.col(label).map_batches(linear_correction, return_dtype=pl.Float64 ).alias(label)
            )

    if ui["checkbox_window"].value:   
        for label in raw_data.columns:
            corrected_data = corrected_data.with_columns(
                pl.col(label).map_batches(windowed_correction, return_dtype=pl.Float64 ).alias(label)
            )

    # normalized_data = raw_data
    # normalized_data
    return corrected_data, raw_data


@app.cell
def _(chart, mo, ui, ui_dropdown):
    mo.vstack([ui_dropdown, ui.hstack(justify= "start"), chart])

    return


@app.cell
def _(io, mo, np, pl, requests, ui_file):
    if ui_file.value:
        file_name = ui_file.value[0].name
        _file = io.BytesIO(ui_file.value[0].contents)
    else:
        file_name = "2014001_C2_03.npz"
        file_path = str(mo.notebook_location() / "public" / file_name)
        if file_path.startswith('http'):
            response  = requests.get(file_path)
            _file = io.BytesIO(response.content)
        else:
            _file = open(file_path, 'rb')

    with np.load(_file) as _npz_file:
        _alldata = dict(_npz_file.items())
    _file.close()


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

    return all_df, file_name


@app.cell
def _(alt, corrected_data, file_name, mo, np, pl, raw_data, sillywalk, ui):


    _n_modes = ui['silder_modes'].value

    fourier_coefficients = sillywalk.anybody.compute_fourier_coefficients(
        corrected_data, _n_modes
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

    nsteps = len(corrected_data)
    reconstructed = reconstruct_signal(
        A, B, nsteps
    )  # ((A @ np.cos(harmonics) + B @ np.sin(harmonics))

    source = pl.DataFrame(
        {
            "steps": np.arange(nsteps),
            "original": raw_data.to_numpy()[:,0],
            "corrected": corrected_data.to_numpy()[:, 0],
            "reconstructed": reconstructed,

        }
    ).unpivot(index="steps", variable_name="Signal")
    # source

    chart = mo.ui.altair_chart(
    alt.Chart(source).mark_line().encode(x="steps", y="value", color="Signal"
        ).properties(
        title=file_name
    ).interactive()
    )

    return (chart,)


@app.cell
def _(all_df, mo):

    _label = all_df.columns[0]

    ui_dropdown = mo.ui.dropdown(options=all_df.columns, value=_label, label="Variable:")

    ui = mo.ui.dictionary(
      {"silder_modes": mo.ui.number(1, 25, value = 4, label="modes"),
       "checkbox_offset": mo.ui.checkbox(label = "Offset correction"),
       "checkbox_linear": mo.ui.checkbox(label="linear correction"),
       "checkbox_window": mo.ui.checkbox(label="Windowed correction"),
       "number_window":  mo.ui.number(0.05,0.5, step=0.05)
      }
    )

    return ui, ui_dropdown


@app.cell
def _():
    return


@app.cell
def _():
    import io
    import altair as alt
    import numpy as np
    import polars as pl
    import requests
    import sillywalk
    from scipy import signal

    return alt, io, np, pl, requests, sillywalk


@app.cell
def _():
    import marimo as mo


    return (mo,)


if __name__ == "__main__":
    app.run()
