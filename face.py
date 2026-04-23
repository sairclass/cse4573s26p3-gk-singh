'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####

    if img.dim() != 3:
        return detection_results

    if img.shape[0] == 3:
        img_hwc = img.permute(1, 2, 0).contiguous()
    else:
        img_hwc = img.contiguous()

    if img_hwc.dtype != torch.uint8:
        img_hwc = img_hwc.to(torch.uint8)

    scale = 1.0
    if img_hwc.shape[0] < 300 or img_hwc.shape[1] < 300:
        scale = 2.0
        img_hwc = torch.nn.functional.interpolate(
            img_hwc.permute(2, 0, 1).unsqueeze(0).float(),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).to(torch.uint8)

    img_np = img_hwc.cpu().numpy()
    face_locations = face_recognition.face_locations(img_np)

    for (top, right, bottom, left) in face_locations:
        x = float(left) / scale
        y = float(top) / scale
        w = float(right - left) / scale
        h = float(bottom - top) / scale
        detection_results.append([x, y, w, h])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####

    if len(imgs) == 0 or K <= 0:
        return cluster_results

    filenames: List[str] = []
    features_list: List[torch.Tensor] = []

    for img_name in imgs:
        img = imgs[img_name]
        img_hwc = to_hwc_uint8(img)
        img_np = img_hwc.cpu().numpy()

        # Reuse Task 1 detector
        detected_boxes = detect_faces(img)
        face_locations = xywh_to_trbl(detected_boxes)

        # Keep only the largest face if multiple are found
        if len(face_locations) > 1:
            face_locations = [largest_box(face_locations)]

        # Use whole image if no detection
        if len(face_locations) == 0:
            H = img_hwc.shape[0]
            W = img_hwc.shape[1]
            face_locations = [(0, W, H, 0)]

        encodings = face_recognition.face_encodings(img_np, face_locations)

        if len(encodings) == 0:
            feature = fallback_feature(img_hwc)
        else:
            feature = torch.tensor(encodings[0], dtype=torch.float32)

        filenames.append(img_name)
        features_list.append(feature)

    features = torch.stack(features_list, dim=0)   # N x D
    features = normalize_features(features)
    labels = kmeans_multi(features, K, num_restarts=5)

    for idx, label in enumerate(labels.tolist()):
        cluster_results[int(label)].append(filenames[idx])

    for cluster in cluster_results:
        cluster.sort()    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

def to_hwc_uint8(img: torch.Tensor) -> torch.Tensor:
    '''
    Convert input image to H x W x 3 uint8 format.
    Args:
        img: input image tensor, can be in any shape and dtype.
    Returns:
        img_hwc: image tensor in H x W x 3 uint8 format.
    '''
    if img.dim() != 3:
        return img

    if img.shape[0] == 3:
        img_hwc = img.permute(1, 2, 0).contiguous()
    else:
        img_hwc = img.contiguous()

    if img_hwc.dtype != torch.uint8:
        img_hwc = img_hwc.to(torch.uint8)

    return img_hwc


def largest_box(face_locations) -> tuple:
    """
    Find the face box with the largest area.
    Args:
        face_locations: list of face boxes in (top, right, bottom, left) format.
    Returns:
        The face box with the largest area.
    """
    best_box = face_locations[0]
    best_area = -1.0

    for (top, right, bottom, left) in face_locations:
        width = max(0, right - left)
        height = max(0, bottom - top)
        area = float(width * height)
        if area > best_area:
            best_area = area
            best_box = (top, right, bottom, left)

    return best_box


def fallback_feature(img_hwc: torch.Tensor) -> torch.Tensor:
    """
    Deterministic fallback feature when face_recognition fails to return an encoding.
    Args:
        img_hwc: input image in H x W x 3 uint8 format.
    Returns:
        A 128-dim feature vector derived from the image content.
    """
    img_f = img_hwc.to(torch.float32) / 255.0
    H = img_f.shape[0]
    W = img_f.shape[1]

    img_chw = img_f.permute(2, 0, 1).unsqueeze(0)

    pooled = torch.nn.functional.adaptive_avg_pool2d(img_chw, (6, 7))
    feat = pooled.reshape(-1)

    mean_val = img_f.mean()
    std_val = img_f.std()
    feat = torch.cat([feat, mean_val.view(1), std_val.view(1)], dim=0)

    return feat.to(torch.float32)


def pairwise_squared_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared Euclidean distance between all pairs of points.
    Args:
        x: torch.Tensor of shape N x D.
        y: torch.Tensor of shape M x D.
    Returns:
        torch.Tensor of shape N x M with squared Euclidean distances.
    """
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1).unsqueeze(0)
    dist = x_norm + y_norm - 2.0 * (x @ y.t())
    return torch.clamp(dist, min=0.0)


def init_centroids(features: torch.Tensor, K: int, seed_offset: int = 0) -> torch.Tensor:
    """
    Initialize centroids for k-means clustering.
    Args:
        features: normalized feature matrix of shape N x D.
        K: number of clusters.
        seed_offset: restart-dependent offset for centroid initialization.
    Returns:
        Initial centroids of shape K x D using deterministic farthest-first selection.
    """
    N = features.shape[0]
    D = features.shape[1]

    if K >= N:
        centroids = features.clone()
        if K > N:
            extra = features[:1].repeat(K - N, 1)
            centroids = torch.cat([centroids, extra], dim=0)
        return centroids

    centroids = torch.zeros((K, D), dtype=features.dtype)

    start_idx = int(seed_offset % N)
    centroids[0] = features[start_idx]

    min_dist = pairwise_squared_dist(features, centroids[0:1]).squeeze(1)

    for k in range(1, K):
        next_idx = int(torch.argmax(min_dist).item())
        centroids[k] = features[next_idx]
        dist_to_new = pairwise_squared_dist(features, centroids[k:k+1]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_to_new)

    return centroids

def xywh_to_trbl(boxes: List[List[float]]) -> List[tuple]:
    """
    Convert bounding boxes from [x, y, width, height] format to (top, right, bottom, left) format.
    Args:
        boxes: list of boxes in [x, y, width, height] format.
    Returns:
        List of boxes in (top, right, bottom, left) format for face_recognition.
    """
    face_locations = []
    for box in boxes:
        x, y, w, h = box
        top = int(round(y))
        left = int(round(x))
        bottom = int(round(y + h))
        right = int(round(x + w))
        face_locations.append((top, right, bottom, left))
    return face_locations


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize the input feature matrix.
    Args:
        features: feature matrix of shape N x D.
    Returns:
        L2-normalized features with the same shape.
    """
    norms = torch.norm(features, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    return features / norms


def kmeans_objective(features: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor) -> float:
    """
    Compute the k-means objective function.
    Args:
        features: feature matrix of shape N x D.
        labels: cluster assignment per sample, shape N.
        centroids: centroid matrix of shape K x D.
    Returns:
        Sum of squared distances from samples to assigned centroids.
    """
    assigned = centroids[labels]
    loss = ((features - assigned) ** 2).sum()
    return float(loss.item())


def kmeans_multi(features: torch.Tensor, K: int, num_restarts: int = 5) -> torch.Tensor:
    """
    Perform k-means clustering with multiple restarts.
    Args:
        features: normalized feature matrix of shape N x D.
        K: number of clusters.
        num_restarts: number of deterministic restarts with different offsets.
    Returns:
        Best label assignment (shape N) by minimum k-means objective.
    """
    best_labels = None
    best_score = None

    for restart_idx in range(num_restarts):
        labels = kmeans(features, K, seed_offset=restart_idx)
        centroids = compute_centroids(features, labels, K)
        score = kmeans_objective(features, labels, centroids)

        if best_score is None or score < best_score:
            best_score = score
            best_labels = labels.clone()

    return best_labels

def compute_centroids(features: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """
    Compute the centroids for each cluster.
    Args:
        features: feature matrix of shape N x D.
        labels: cluster assignment per sample, shape N.
        K: number of clusters.
    Returns:
        Centroid matrix of shape K x D.
    """
    D = features.shape[1]
    centroids = torch.zeros((K, D), dtype=features.dtype)

    for k in range(K):
        mask = (labels == k)
        if mask.any():
            centroids[k] = features[mask].mean(dim=0)
        else:
            centroids[k] = features[0]

    return centroids


def kmeans(features: torch.Tensor, K: int, max_iters: int = 100, seed_offset: int = 0) -> torch.Tensor:
    """
    Perform k-means clustering with a single initialization.
    Args:
        features: normalized feature matrix of shape N x D.
        K: number of clusters.
        max_iters: maximum number of k-means iterations.
        seed_offset: restart-dependent offset for centroid initialization.
    Returns:
        Cluster labels (shape N) with values in [0, K-1].
    """
    N = features.shape[0]

    if N == 0:
        return torch.zeros((0,), dtype=torch.long)

    if K == 1:
        return torch.zeros((N,), dtype=torch.long)

    centroids = init_centroids(features, K, seed_offset=seed_offset)
    labels = torch.full((N,), -1, dtype=torch.long)

    for _ in range(max_iters):
        dists = pairwise_squared_dist(features, centroids)
        new_labels = torch.argmin(dists, dim=1)

        if torch.equal(new_labels, labels):
            break

        labels = new_labels
        new_centroids = centroids.clone()

        for k in range(K):
            mask = (labels == k)
            if mask.any():
                new_centroids[k] = features[mask].mean(dim=0)
            else:
                assigned_centroids = centroids[labels]
                point_errors = ((features - assigned_centroids) ** 2).sum(dim=1)
                farthest_idx = int(torch.argmax(point_errors).item())
                new_centroids[k] = features[farthest_idx]

        shift = ((new_centroids - centroids) ** 2).sum()
        centroids = new_centroids

        if float(shift.item()) < 1e-8:
            break

    dists = pairwise_squared_dist(features, centroids)
    labels = torch.argmin(dists, dim=1)
    return labels
