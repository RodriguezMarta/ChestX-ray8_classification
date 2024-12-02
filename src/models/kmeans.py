from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import cupy as cp 
def train_kmeans(data, n_clusters=14):
    if cp.is_available():
        data_gpu = cp.asarray(data)  # Transfer data to GPU memory
        minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
        minibatch_kmeans.fit(data_gpu)  # Train on GPU
        return minibatch_kmeans.get_params()  # Return model parameters

    else:
        print("GPU not detected. Using CPU for MiniBatchKMeans training.")
        minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
        minibatch_kmeans.fit(data)
        return minibatch_kmeans


def evaluate_kmeans(model, data):
    labels = model.predict(data)

    # Use cuPy if model was trained on GPU
    if isinstance(model.cluster_centers_, cp.core.core.ndarray):
        inertia = cp.asnumpy(model.inertia_)  # Convert inertia from GPU to CPU if applicable
    else:
        inertia = model.inertia_

    return labels, inertia