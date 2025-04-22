import numpy as np

import workflow.util.seismo_reader as seismo_reader


def seismo_fft(
    collection: seismo_reader.SeismogramCollection, periodify: bool = True
) -> seismo_reader.SeismogramCollection:
    def transform(series: np.ndarray | None, seistype, station) -> np.ndarray | None:
        if series is None:
            return None
        N = series.shape[0]
        dt = (series[-1, 0] - series[0, 0]) / (N - 1)
        largest_dt_deviation = np.max(np.abs(series[1:, 0] - series[:-1, 0] - dt))
        if largest_dt_deviation > 1e-2 * dt and largest_dt_deviation > 1e-4:
            raise ValueError(
                f"Attempting to perform an fft on seistype {seistype} at station {station} with non-uniform data.\n{series[1:, 0] - series[:-1, 0]} (should be {dt})"
            )
        if periodify:
            return np.stack(
                [
                    (2 * np.pi)
                    * np.concatenate([np.arange(N - 1), np.arange(-N + 1, 0)], axis=0),
                    np.fft.fft(
                        np.concatenate(
                            [series[:, 1], np.flip(series[1:-1, 1])], axis=0
                        ),
                        norm="forward",
                    ),
                ],
                axis=1,
            )
        else:
            return np.stack(
                [
                    (2 * np.pi / (dt * N)) * np.arange(N),
                    np.fft.fft(series[:, 1], norm="forward"),
                ],
                axis=1,
            )

    return collection.copy_with_datachange(
        _seismos={
            t: [transform(s, t, i) for i, s in enumerate(v)]
            for t, v in collection._seismos.items()
        }
    )
