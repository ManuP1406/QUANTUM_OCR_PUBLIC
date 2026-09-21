import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import cv2
from imutils import resize
from skimage.morphology import skeletonize
import PIL


def plot_graph_attention(nodes, edge_index, node_weights=None, node_size=60, cmap='GnBu', node_shape='o', edge_width=2.0,
                         edge_alpha=1.):
    if node_weights is None:
        node_weights = [0.5] * len(nodes)

    G = nx.Graph()
    for i, (pos, weight) in enumerate(zip(nodes, node_weights)):
        G.add_node(i, pos=(pos[0], pos[1]))

    for edge in edge_index:
        G.add_edge(*edge)

    g_nodes = nx.draw_networkx_nodes(G, nodes, cmap=plt.get_cmap(cmap), node_color=node_weights,
                                     node_size=node_size, node_shape=node_shape)
    g_nodes.set_edgecolor('k')
    nx.draw_networkx_edges(G, nodes, alpha=edge_alpha, width=edge_width)
    plt.axis('on')


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def spatial_cluster(points, d, cluster_with_center=False):
    # Groups points that are spatially close to each other (within distance 'd').
    # This is a simple greedy clustering algorithm.
    clusters = []
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            cluster = [i]
            cluster_center = np.array(points[i]).astype(float)
            taken[i] = True
            for j in range(i+1, n):
                if cluster_with_center:
                    base_point = cluster_center
                else:
                    base_point = points[i]
                
                # If point 'j' is close enough to the cluster center (or the first point), add it to the cluster
                if dist(base_point, points[j]) < d:
                    # Update cluster center running average
                    cluster_center = (cluster_center * len(cluster) + np.array(points[j])) / (len(cluster) + 1)
                    cluster.append(j)
                    taken[j] = True
            clusters.append(cluster)

    return clusters


def spatial_fuse(points, d):
    # Replaces points in each cluster with the average coordinate of that cluster
    clusters = spatial_cluster(points, d)
    ret = copy.copy(points).astype(np.float32)
    for cluster in clusters:
        avg_coord = points[cluster].mean(0)
        ret[cluster] = avg_coord
    return np.array(ret)


def _cal_edge(id1, id2):
    return [id1, id2] if id1 < id2 else [id2, id1]


def merge_close_nodes(nodes, edge_index, node_vals, merge_thres, merge_with_center):
    # Simplifies the graph by merging nodes that are very close together.
    # 1. Cluster nodes based on distance 'merge_thres'
    nodes = np.array(nodes).astype(np.float32)
    clusters = spatial_cluster(nodes, merge_thres, merge_with_center)
    map_id = {}
    new_nodes = []
    new_node_vals = []

    # 2. Create new nodes from cluster centers
    for i, cluster in enumerate(clusters):
        for id in cluster:
            map_id[id] = i  # Map old node ID to new cluster ID

        new_nodes.append(nodes[cluster].mean(0))
        new_node_vals.append(node_vals[cluster].mean())

    # 3. Rewire edges to new cluster IDs
    new_edge_index = []
    for edge in edge_index:
        id0 = map_id[edge[0]]
        id1 = map_id[edge[1]]

        # Ignore self-loops (edges within the same cluster)
        if id0 == id1:
            continue

        edge = _cal_edge(id0, id1)
        if edge not in new_edge_index:
            new_edge_index.append(edge)

    return np.array(new_nodes), new_edge_index, np.array(new_node_vals)


def construct_init_dense_graph(map, img, half_win_size=1):
    # 1. Identify all non-zero pixels in the skeleton map
    # 'map' is the skeletonized binary image. 'pts' will be a list of [row, col] coordinates of the skeleton.
    pts = np.stack(np.where(map > 0)).transpose().astype(int)

    # ... (debug visualization code commented out) ...

    # Helper to clamp coordinate within image boundaries [0, height-1] or [0, width-1]
    def _valid_pos(val):
        return max(min(int(abs(val)), map.shape[0] - 1), 0)

    # 2. Assign values to each node (skeleton point)
    # For every skeleton point (pt), extract a small window around it from the ORIGINAL image (img)
    # The size of the window is (2*half_win_size + 1)
    # Then calculate the MEAN pixel intensity of that window.
    # This assigns a value to each node based on the brightness of the stroke at that point.
    node_vals = [img[_valid_pos(pt[0] - half_win_size): _valid_pos(pt[0] + half_win_size + 1),
                 _valid_pos(pt[1] - half_win_size): _valid_pos(pt[1] + half_win_size + 1)].mean()
                 for pt in pts]

    node_vals = np.array(node_vals, dtype=np.float32)

    # Helper to hash coordinates to a single integer for dictionary lookup
    def hash(pt):
        return pt[0] * map.shape[0] + pt[1]

    # 3. Create a mapping from coordinate hash to node index
    node_id_dict = {}
    for i, pt in enumerate(pts):
        node_id_dict[hash(pt)] = i

    # Helper to get the node index from a coordinate point
    def _calc_pid(pt):
        return node_id_dict[hash(pt)]

    # Helper to check if a point is valid (within bounds AND is part of the skeleton)
    def _is_valid_pt(pt):
        if 0 <= pt[0] < map.shape[0] and \
                0 <= pt[1] < map.shape[1] and \
                map[pt[0], pt[1]] > 0:
            return True
        return False

    edges = []
    # 4. Connect 4-connected neighbors (Up, Down, Left, Right)
    for pt in pts:
        pt_id = _calc_pid(pt)
        for i, j in [(0, 1), [1, 0], [0, -1], [-1, 0]]:
            new_pt = [pt[0] + i, pt[1] + j]
            if _is_valid_pt(new_pt):
                # Add edge between current point and its neighbor
                edges.append(_cal_edge(pt_id, _calc_pid(new_pt)))

    # 5. Connect diagonal neighbors (Top-Left, Top-Right, etc.)
    # But ONLY if they are not already connected via a cardinal neighbor (to avoid "crossing" edges in tight corners)
    for pt in pts:
        pt_id = _calc_pid(pt)
        for i, j in [(1, 1), [1, -1], [-1, -1], [-1, 1]]:
            add_cx_edge = True
            cx_nb_pt = [pt[0] + i, pt[1] + j]
            if _is_valid_pt(cx_nb_pt):
                # Check adjacent cardinal points to see if we should skip this diagonal connection
                candi_pt1 = [pt[0] + i, pt[1]]
                candi_pt2 = [pt[0], pt[1] + j]
                for candi_pt in [candi_pt1, candi_pt2]:
                    if _is_valid_pt(candi_pt):
                        add_cx_edge = False
                        break

                if add_cx_edge:
                    edges.append(_cal_edge(pt_id, _calc_pid(cx_nb_pt)))

    return pts.astype(float), edges, node_vals


def _get_neibors(pid, bi_edges):
    eid_of_pid = np.where(bi_edges[:, 0] == pid)[0]
    return bi_edges[eid_of_pid, 1]


def calc_k_order_edges(bi_edges, bi_k_1_edges, num_nodes=None):
    '''edge_kth = connect(edge_k_1, edge_1)'''
    if num_nodes is None:
        num_nodes = bi_edges.max() + 1

    # 宽度优先
    edges_kth_orders = []
    for pid in range(num_nodes):
        neibors_k_1_order = _get_neibors(pid, bi_k_1_edges)
        for nb_k_1_pid in neibors_k_1_order:
            neibors_k_order = _get_neibors(nb_k_1_pid, bi_edges)
            for nb_kth_pid in neibors_k_order:
                if nb_kth_pid != pid:
                    new_edge = _cal_edge(nb_kth_pid, pid)
                    edges_kth_orders.append(new_edge)

    return edges_kth_orders


def _bidirection_edges(edges):
    edges_ndarr = np.array(edges).astype(int)
    bi_edges = np.concatenate([edges_ndarr, edges_ndarr[:, [1, 0]]])
    return bi_edges


def remove_small_node_cluster(pts, edges, node_vals):
    '''to be done'''
    pt_idxs = set(range(len(node_vals)))
    bi_edges = _bidirection_edges(edges)
    clusters = []

    while(len(pt_idxs) > 0):
        pid = pt_idxs.pop()

        cluster = [pid]
        neibors_k_1_order = _get_neibors(pid, bi_edges)
        cluster += neibors_k_1_order
        break


import numpy as np
import math

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle formed by three points p1-p2-p3 (with p2 as the vertex).
    Returns the angle in radians.
    """
    # Vectors from p2 to p1 and from p2 to p3
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # Normalize vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    v1_norm = v1 / norm_v1
    v2_norm = v2 / norm_v2
    
    # Calculate angle using dot product
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    return angle

def remove_collinear_nodes(nodes, edges, node_vals, angle_threshold_deg=5, merge_dist=0.1):
    nodes = list(nodes)
    node_vals = list(node_vals)
    angle_threshold = np.radians(angle_threshold_deg)

    adj = {i: set() for i in range(len(nodes))}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    changed = True
    while changed:
        changed = False
        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            neighbors = [n for n in adj[i] if nodes[n] is not None]

     
            if len(neighbors) == 2:
                n1, n2 = neighbors
                ang = calculate_angle(nodes[n1], nodes[i], nodes[n2])
                if ang > (np.pi - angle_threshold):
                    adj[n1].discard(i)
                    adj[n2].discard(i)
                    adj[i] = set()
                    if n2 not in adj[n1]:
                        adj[n1].add(n2)
                        adj[n2].add(n1)
                    nodes[i] = None
                    node_vals[i] = None
                    changed = True
                    break

          
            if merge_dist is not None:
                for n in neighbors:
                    d = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[n]))
                    if d < merge_dist:
                   
                        new_pos = (np.array(nodes[i]) + np.array(nodes[n])) / 2
                        new_val = (node_vals[i] + node_vals[n]) / 2
                        nodes[i] = new_pos.tolist()
                        node_vals[i] = new_val

                     
                        for nb in list(adj[n]):
                            if nb == i:
                                continue
                            adj[nb].discard(n)
                            adj[nb].add(i)
                            adj[i].add(nb)

                        adj[i].discard(n)
                        adj[n] = set()
                        nodes[n] = None
                        node_vals[n] = None
                        changed = True
                        break

                if changed:
                    break

   
    old_to_new = {}
    new_nodes, new_node_vals = [], []
    for i, node in enumerate(nodes):
        if node is not None:
            old_to_new[i] = len(new_nodes)
            new_nodes.append(node)
            new_node_vals.append(node_vals[i])

    new_edges = []
    for u in old_to_new:
        for v in adj[u]:
            if v in old_to_new and old_to_new[u] < old_to_new[v]:
                new_edges.append((old_to_new[u], old_to_new[v]))

    return np.array(new_nodes), new_edges, np.array(new_node_vals)

def construct_graph(map, img, half_win_size=1, merge_dist_thres=0., isolate_num_thresh=2, 
                   edge_order=1, merge_with_center=False, sort_by_geomtric=True,
                   collinear_removal=True, angle_threshold_deg=25):
    """
    Enhanced graph construction with collinear node removal.
    """
    pts, edges, node_vals = construct_init_dense_graph(map, img, half_win_size)

    if merge_dist_thres > 0.99:
        pts, edges, node_vals = merge_close_nodes(pts, edges, node_vals, merge_dist_thres, merge_with_center)
    
    # Add collinear node removal
    if collinear_removal:
        pts, edges, node_vals = remove_collinear_nodes(pts, edges, node_vals, angle_threshold_deg)

    node_vals /= 255.
    pts = pts[:, [1, 0]]
    pts[:, 1] = map.shape[0] - pts[:, 1] - 1
    pts = pts * 2 / map.shape[0] - 1

    if sort_by_geomtric:
        pts = [tuple(pt) for pt in pts]
        pts = np.array(pts, dtype=[('x', '<f8'), ('y', '<f8')])
        dict_id = {}
        ind = np.argsort(pts, order=('x', 'y'))
        new_pts, new_node_vals = [], []
        for i, id in enumerate(ind):
            dict_id[id] = i
            new_pts.append(list(pts[id]))
            new_node_vals.append(node_vals[id])
        pts = np.array(new_pts, dtype=np.float32)
        node_vals = np.array(new_node_vals, dtype=np.float32)

        edges = np.array(edges, dtype=int)
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                edges[i, j] = dict_id[edges[i, j]]

    return pts, edges, node_vals


def image2graph(image, width, ref_img=None, with_processing=False, debug=False, **kwargs):
    # --- Handle different input types (file path, PIL image, or numpy array) ---
    if isinstance(image, str):
        # Load from file path
        img = cv2.imread(image, -1)
    elif isinstance(image, PIL.Image.Image):
        # Convert PIL image to numpy array (RGB -> BGR -> Gray)
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif isinstance(image, np.ndarray):
        # Already a numpy array, use as is
        img = image

   
    if width != img.shape[1]:
        img = resize(img, width=width)

    
    if ref_img is None:
        org_img = img
    else:
        org_img = ref_img

    if with_processing:
        pass

    # enhance contrast  
    img = cv2.equalizeHist(img)
    # binarize
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # skeletonize
    thin_img = skeletonize(binary / 255.)

    nodes, edges, node_vals = construct_graph(thin_img, org_img, **kwargs)
    # construct_2nd_order_graph

    if debug:
        print(thin_img.shape)
        plt.figure(figsize=(5, 5))
        plt.subplot(221)
        plt.imshow(org_img, cmap='gray')

        plt.subplot(222)
        plt.imshow(binary, cmap='gray')

        plt.subplot(223)
        plt.imshow(thin_img, cmap='gray')

        plt.subplot(224)
        plot_graph_attention(nodes, edges, node_vals * 255.)
        print('node_vals:', node_vals)
        print('#nodes: %d  #edges: %d' % (len(nodes), len(edges)))
        print('val min: %.2f  max: %.2f'%(node_vals.min(), node_vals.max()))

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.tight_layout()
        plt.show()

    return nodes, edges, node_vals


def plot_graph_img(nodes, edges, image_size=64):
    nodes = (nodes * image_size + image_size) / 2
    nodes = nodes.astype(int).clip(0, image_size - 1)
    nodes[:, 1] = image_size - nodes[:, 1]
    img = np.zeros((image_size, image_size))
    for id1, id2 in edges:
        cv2.line(img, tuple(nodes[id1]), tuple(nodes[id2]), 255, 1)
    return img
