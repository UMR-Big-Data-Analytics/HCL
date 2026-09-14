import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

from vmf_hac.definitions import ROOT_DIR


def lat_lon_to_cartesian(lat, lon, R=1.0):
    """Converts arrays of Latitude and Longitude to 3D Cartesian Coordinates"""
    lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
    phi = (90 - lat) * np.pi / 180
    theta = lon * np.pi / 180
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]) * R


def get_camera_vec(elev, azim):
    """Calculates the exact 3D vector of the Matplotlib camera"""
    elev_rad, azim_rad = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(elev_rad) * np.cos(azim_rad), np.cos(elev_rad) * np.sin(azim_rad), np.sin(elev_rad)])


def mask_hidden_points(pts_3d, camera_vec, threshold=-0.05):
    """Returns a boolean mask of points on the visible hemisphere"""
    return np.dot(pts_3d, camera_vec) > threshold


def get_real_earth_population(n_samples=1500):
    """Downloads real global city populations and samples points based on density"""
    print("Downloading real population data (Natural Earth 50m)...")
    url_cities = (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_populated_places.geojson"
    )
    gdf_cities = gpd.read_file(url_cities)

    # Filter valid populations
    cities = gdf_cities[["geometry", "POP_MAX"]].dropna()
    cities = cities[cities["POP_MAX"] > 0]

    # Weight probabilities by population
    probs = cities["POP_MAX"] / cities["POP_MAX"].sum()  # noqa

    print(f"Sampling {n_samples} individuals weighted by global population...")
    sampled = cities.sample(n=n_samples, replace=True, random_state=42)

    # lons = cities.geometry.x.to_numpy(copy=True)
    # lats = cities.geometry.y.to_numpy(copy=True)

    lons = sampled.geometry.x.to_numpy(copy=True)
    lats = sampled.geometry.y.to_numpy(copy=True)

    # Add slight Gaussian noise (~0.5 deg) so megalopolises don't collapse into a single mathematical point
    # (vMF HAC requires gamma to handle perfectly identical points. Noise provides natural continuous variance).
    np.random.seed(42)
    lons += np.random.normal(0, 0.5, size=len(lons))
    lats += np.random.normal(0, 0.5, size=len(lats))
    lats = np.clip(lats, -89.9, 89.9)

    return lat_lon_to_cartesian(lats, lons, R=1.001)


# ==========================================
# 3. HIGH-RESOLUTION DOTTED GLOBE MAP
# ==========================================
def create_dotted_globe(n_points=12000):
    """Generates a dense Fibonacci sphere and intersects it with real landmasses"""
    print("Generating vector map layers...")

    # 1. Fibonacci Sphere
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    pts_3d = np.column_stack([x, y, z])

    lats = 90 - (phi * 180 / np.pi)
    lons = theta * 180 / np.pi % 360
    lons[lons > 180] -= 360

    df = pd.DataFrame({"lat": lats, "lon": lons})
    gdf_pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")  # ty:ignore

    # 2. Intersect with Land
    url_land = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson"
    land = gpd.read_file(url_land)

    print("Computing spatial join for map rendering...")
    pts_in_land = gpd.sjoin(gdf_pts, land, how="inner", predicate="intersects")
    land_mask = df.index.isin(pts_in_land.index)

    land_pts_3d = pts_3d[land_mask]

    # 3. Densify Coastlines
    coastlines = []
    for geom in land.geometry:
        if geom is None:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else geom.geoms
        for poly in polys:
            lon_c, lat_c = poly.exterior.coords.xy
            lon_c, lat_c = np.array(lon_c), np.array(lat_c)

            dense_lon, dense_lat = [], []
            for i in range(len(lon_c) - 1):
                dist = np.sqrt((lon_c[i + 1] - lon_c[i]) ** 2 + (lat_c[i + 1] - lat_c[i]) ** 2)
                steps = max(2, int(dist * 2.0))
                dense_lon.extend(np.linspace(lon_c[i], lon_c[i + 1], steps)[:-1])
                dense_lat.extend(np.linspace(lat_c[i], lat_c[i + 1], steps)[:-1])
            dense_lon.append(lon_c[-1])
            dense_lat.append(lat_c[-1])
            coastlines.append(lat_lon_to_cartesian(np.array(dense_lat), np.array(dense_lon), R=1.000))

    return land_pts_3d, coastlines


def custom_vmf_hac(X, k, gamma):
    N = X.shape[0]
    clusters = {i: [i] for i in range(N)}

    sums = np.zeros((2 * N, 3))
    sums[:N] = X
    Ns = np.zeros(2 * N)
    Ns[:N] = 1.0
    Rs = np.zeros(2 * N)
    Rs[:N] = 1.0
    D = np.full((2 * N, 2 * N), np.inf)

    for i in range(N):
        S_ij = sums[i] + sums[i + 1 : N]
        R_ij = np.linalg.norm(S_ij, axis=1)
        N_ij = Ns[i] + Ns[i + 1 : N]
        D[i, i + 1 : N] = (
            N_ij * np.log(1 - R_ij / N_ij + gamma)
            - Ns[i] * np.log(1 - Rs[i] / Ns[i] + gamma)
            - Ns[i + 1 : N] * np.log(1 - Rs[i + 1 : N] / Ns[i + 1 : N] + gamma)
        )

    active = set(range(N))
    next_id = N

    while len(active) > k:
        act_arr = np.array(list(active))
        sub_D = D[np.ix_(act_arr, act_arr)]
        min_idx = np.unravel_index(np.argmin(sub_D), sub_D.shape)
        u, v = act_arr[min_idx[0]], act_arr[min_idx[1]]
        if u > v:
            u, v = v, u

        sums[next_id] = sums[u] + sums[v]
        Ns[next_id] = Ns[u] + Ns[v]
        Rs[next_id] = np.linalg.norm(sums[next_id])
        clusters[next_id] = clusters[u] + clusters[v]

        active.remove(u)
        active.remove(v)

        act_list = list(active)
        if act_list:
            S_ia = sums[next_id] + sums[act_list]
            R_ia = np.linalg.norm(S_ia, axis=1)
            N_ia = Ns[next_id] + Ns[act_list]

            dists = (
                N_ia * np.log(1 - R_ia / N_ia + gamma)
                - Ns[act_list] * np.log(1 - Rs[act_list] / Ns[act_list] + gamma)
                - Ns[next_id] * np.log(1 - Rs[next_id] / Ns[next_id] + gamma)
            )

            for idx_a, a in enumerate(act_list):
                D[min(next_id, a), max(next_id, a)] = dists[idx_a]

        D[u, :] = np.inf
        D[:, u] = np.inf
        D[v, :] = np.inf
        D[:, v] = np.inf
        active.add(next_id)
        next_id += 1

    labels = np.zeros(N, dtype=int)
    for lbl, cluster_id in enumerate(active):
        for pt in clusters[cluster_id]:
            labels[pt] = lbl
    return labels


def align_labels(labels_ref, labels_target):
    k = max(labels_ref.max(), labels_target.max()) + 1
    cost_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost_matrix[i, j] = -np.sum((labels_ref == i) & (labels_target == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {col: row for row, col in zip(row_ind, col_ind, strict=False)}
    return np.array([mapping[label] for label in labels_target])


if __name__ == "__main__":
    X_earth = get_real_earth_population(2000)
    land_pts_3d, coastlines = create_dotted_globe()

    K_CLUSTERS = 28
    gammas = [0.01, 0.001]

    print("Clustering Ward baseline...")
    ward = AgglomerativeClustering(n_clusters=K_CLUSTERS, linkage="ward")
    labels_ward = ward.fit_predict(X_earth)

    labels_vmf = {}
    for g in gammas:
        print(f"Clustering vMF with gamma={g}...")
        lbls = custom_vmf_hac(X_earth, K_CLUSTERS, g)
        labels_vmf[g] = align_labels(labels_ward, lbls)

    print("Rendering high-res visualization...")
    views = [
        ("Europe \\& Africa", 35, 15),
    ]

    # Margins are given in inches so the figure height follows the globe size
    # instead of being inflated by fractional margins.
    n_rows = len(views)
    n_cols = len(gammas) + 1
    spacing_in = 0.15
    fig_w = 8.0
    side_in = 0.02  # left/right margin
    title_in = 0.45  # room for the two-line column titles
    bottom_in = 0.02
    avail_w = fig_w - 2 * side_in
    ax_side = (avail_w - spacing_in * (n_cols - 1)) / n_cols
    avail_h = ax_side * n_rows + spacing_in * (n_rows - 1)
    fig_h = avail_h + title_in + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    cluster_colors = plt.get_cmap("tab20", K_CLUSTERS)(np.arange(K_CLUSTERS))
    wspace = hspace = spacing_in / ax_side
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        left=side_in / fig_w,
        right=1 - side_in / fig_w,
        top=1 - title_in / fig_h,
        bottom=bottom_in / fig_h,
        wspace=wspace,
        hspace=hspace,
    )

    # The sphere is enlarged inside each axes via the box aspect zoom, so no
    # extra axes overlap is needed.
    AXES_SCALE = 1.0
    GLOBE_ZOOM = 1.6

    def _expand(ax):
        p = ax.get_position()
        cx, cy = p.x0 + p.width / 2, p.y0 + p.height / 2
        w, h = p.width * AXES_SCALE, p.height * AXES_SCALE
        ax.set_position([cx - w / 2, cy - h / 2, w, h])
        ax.patch.set_alpha(0)

    for i_v, (_, elev, azim) in enumerate(views):
        cam_vec = get_camera_vec(elev, azim)

        # Determine visibility masks
        vis_data_mask = mask_hidden_points(X_earth, cam_vec)
        vis_land_mask = mask_hidden_points(land_pts_3d, cam_vec, threshold=0.0)

        X_visible = X_earth[vis_data_mask]
        labels_ward_vis = labels_ward[vis_data_mask]
        land_visible = land_pts_3d[vis_land_mask]

        def render_map_context(ax):
            """Renders the perfectly clipped dotted map and coastlines"""
            ax.set_box_aspect([1, 1, 1], zoom=GLOBE_ZOOM)
            ax.set_proj_type("ortho")
            ax.set_axis_off()
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-1.02, 1.02)
            ax.set_zlim(-1.02, 1.02)

            # Globe outline aligned with the current camera view
            if np.linalg.norm(cam_vec) > 0:  # noqa: B023
                cam = cam_vec / np.linalg.norm(cam_vec)  # noqa: B023
                ref = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(cam, ref)) > 0.95:
                    ref = np.array([0.0, 1.0, 0.0])
                u = np.cross(cam, ref)
                u = u / np.linalg.norm(u)
                v = np.cross(cam, u)
                t = np.linspace(0, 2 * np.pi, 256)
                ring = 1.01 * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
                ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="black", lw=0.45, alpha=0.28, zorder=0)

            # Plot dotted land (highly visible background)
            ax.scatter(
                land_visible[:, 0],  # noqa: B023
                land_visible[:, 1],  # noqa: B023
                land_visible[:, 2],  # noqa: B023
                color="#cbd5e1",
                s=2,
                alpha=0.9,
                edgecolors="none",
                zorder=-2,
            )

            # Plot clipped coastlines
            for line in coastlines:
                vis_line_mask = mask_hidden_points(line, cam_vec, threshold=-0.02)  # noqa: B023
                line_clipped = line.copy()
                line_clipped[~vis_line_mask] = np.nan  # NaN breaks the line so it doesn't wrap back
                ax.plot(
                    line_clipped[:, 0],
                    line_clipped[:, 1],
                    line_clipped[:, 2],
                    color="black",
                    lw=0.45,
                    alpha=0.75,
                    zorder=-1,
                )

        # ---- WARD PLOT ----
        ax = fig.add_subplot(gs[i_v, 0], projection="3d")
        _expand(ax)
        render_map_context(ax)
        ax.scatter(
            X_visible[:, 0],
            X_visible[:, 1],
            X_visible[:, 2],
            color=cluster_colors[labels_ward_vis],
            s=10,
            alpha=0.98,
            edgecolors="white",
            linewidth=0.3,
            depthshade=False,
            zorder=1,
        )
        ax.view_init(elev=elev, azim=azim)

        if i_v == 0:
            ax.set_title("Ward", fontsize=16, pad=16)
        # ax.text2D(0.0, 0.5, view_name, transform=ax.transAxes, fontsize=16, rotation=90, va='center', ha='right', weight='bold')

        # ---- vMF PLOTS ----
        for i_g, g in enumerate(gammas):
            labels_vmf_vis = labels_vmf[g][vis_data_mask]

            ax = fig.add_subplot(gs[i_v, i_g + 1], projection="3d")
            _expand(ax)
            render_map_context(ax)
            ax.scatter(
                X_visible[:, 0],
                X_visible[:, 1],
                X_visible[:, 2],
                color=cluster_colors[labels_vmf_vis],
                s=10,
                alpha=0.98,
                edgecolors="white",
                linewidth=0.3,
                depthshade=False,
                zorder=1,
            )
            ax.view_init(elev=elev, azim=azim)

            if i_v == 0:
                ax.set_title(f"HCL\n$\\gamma={g}$", fontsize=16, pad=4)

    plt.savefig(
        ROOT_DIR / "results" / "plots" / "global_population_real_data_europe.pdf",
        dpi=250,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )
