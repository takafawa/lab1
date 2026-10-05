"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    res = np.einsum('pij,pjk->ik', matrices, vectors)
    return res


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    matrix = np.asarray(matrix)
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []
    for row in matrix:
        _, indices = np.unique(row, return_index=True)
        unique_vals = [float(x) for x in row[np.sort(indices)]]
        result.append(unique_vals)
    return result


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []
    for col in matrix.T:
        _, indices = np.unique(col, return_index=True)
        unique_vals = [float(x) for x in col[np.sort(indices)]]
        result.append(unique_vals)
    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed

    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)
    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)

    stats = MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances
    )
    return stats


def plot_matrix_histograms(stats: MatrixStatistics) -> None:
    "Функция для построения гистограмм по строкам и столбцам (требование задачи 4)."
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(stats.row_means, bins=10, alpha=0.7, color='blue', label='Средние по строкам')
    axes[0].set_title("Распределение средних по строкам")
    axes[0].legend()

    axes[1].hist(stats.column_means, bins=10, alpha=0.7, color='green', label='Средние по столбцам')
    axes[1].set_title("Распределение средних по столбцам")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second

    r_idx, c_idx = np.indices((rows, columns))
    result = np.where((r_idx + c_idx) % 2 == 0, first, second)
    return result


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    img[:] = background_color
    cy, cx = image_height / 2.0, image_width / 2.0
    y_min = int(round(cy - height / 2.0))
    y_max = int(round(cy + height / 2.0))
    x_min = int(round(cx - width / 2.0))
    x_max = int(round(cx + width / 2.0))
    y_min, y_max = max(0, y_min), min(image_height, y_max)
    x_min, x_max = max(0, x_min), min(image_width, x_max)
    img[y_min:y_max, x_min:x_max] = shape_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    img[:] = background_color
    cy, cx = (image_height - 1) / 2.0, (image_width - 1) / 2.0
    y_indices, x_indices = np.indices((image_height, image_width))
    mask = ((x_indices - cx) ** 2) / (semi_axis_x ** 2) + ((y_indices - cy) ** 2) / (semi_axis_y ** 2) <= 1.0
    img[mask] = shape_color
    return img


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.asarray(data.values), data.window
    mean_val = float(np.mean(values))
    var_val = float(np.var(values))
    std_val = float(np.std(values))

    if len(values) > 2:
        left = values[:-2]
        center = values[1:-1]
        right = values[2:]
        max_mask = (center > left) & (center > right)
        min_mask = (center < left) & (center < right)
        local_maxima = np.where(max_mask)[0] + 1
        local_minima = np.where(min_mask)[0] + 1
    else:
        local_maxima = np.array([], dtype=int)
        local_minima = np.array([], dtype=int)

    kernel = np.ones(window) / window
    moving_avg = np.convolve(values, kernel, mode='valid')

    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std=std_val,
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_avg
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    class_count = data.class_count

    if class_count is None:
        class_count = int(np.max(labels)) + 1 if len(labels) > 0 else 0

    res = np.zeros((len(labels), class_count), dtype=int)

    if len(labels) > 0 and class_count > 0:
        res[np.arange(len(labels)), labels] = 1

    return res
