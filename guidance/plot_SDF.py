# This cell will try to find an SDF CSV in /mnt/data automatically.
# If there is exactly one .csv, it will use it. Otherwise, set `csv_path` below.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plt_SDF(df):
    if df is not None and df.shape[1] >= 4:
        # Prefer first 4 columns
        x = df.iloc[:,0].to_numpy(dtype=float, copy=False)
        y = df.iloc[:,1].to_numpy(dtype=float, copy=False)
        z = df.iloc[:,2].to_numpy(dtype=float, copy=False)
        sdf = df.iloc[:,3].to_numpy(dtype=float, copy=False)
        n = x.size
        print(f"Loaded {n} points. Columns used: [0:x, 1:y, 2:z, 3:sdf]")
    else:
        x=y=z=sdf=None

    # 4) Downsample if too many points for plotting
    max_points = 4096
    if x is not None and x.size > max_points:
        idx = np.random.RandomState(42).choice(x.size, size=max_points, replace=False)
        x,y,z,sdf = x[idx], y[idx], z[idx], sdf[idx]
        print(f"Downsampled to {max_points} points for plotting.")

    # 5) 3D scatter colored by SDF
    if x is not None:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=sdf, s=1, alpha=0.8, linewidths=0)
        cb = plt.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label("SDF value")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("SDF 3D scatter (颜色表示 SDF 值)")
        # Equal aspect
        ranges = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])
        max_range = ranges.max() if np.isfinite(ranges).all() else 1.0
        x_m = 0.5*(x.max()+x.min())
        y_m = 0.5*(y.max()+y.min())
        z_m = 0.5*(z.max()+z.min())
        ax.set_xlim(x_m - max_range/2, x_m + max_range/2)
        ax.set_ylim(y_m - max_range/2, y_m + max_range/2)
        ax.set_zlim(z_m - max_range/2, z_m + max_range/2)
        plt.show()

    # 6) Optional: show SDF histogram
    if sdf is not None:
        plt.figure(figsize=(7,4))
        plt.hist(sdf, bins=100)
        plt.xlabel("SDF value")
        plt.ylabel("Count")
        plt.title("SDF 直方图")
        plt.show()


if __name__ == "__main__":
    # 1) Locate CSV
    csv_path = "Path/to/sdf_data.csv"

    print(f"Using CSV path: {csv_path}")
    if not os.path.exists(csv_path):
        print("⚠️ CSV didn't found!")

    # 2) Load CSV (assume: col0=x, col1=y, col2=z, col3=sdf). Try robust parsing.
    df = None
    if os.path.exists(csv_path):
        try:
            # First try: no header
            df = pd.read_csv(csv_path, header=None)
            # part1 = df.iloc[-47000:-20000, -4:]  # 列D/E/F/G
            # part2 = df.iloc[:20000, -4:]         # 列D/E/F/G

            # df = pd.concat([part1, part2], axis=0)

            # Ensure at least 4 columns
            if df.shape[1] < 4:
                raise ValueError("Less than 4 columns without header; retry with header inference.")
        except Exception as e:
            # Retry with default header
            df = pd.read_csv(csv_path)
            # Try to locate suitable columns
            if df.shape[1] < 4:
                raise
    
    # 3) Plot
    plt_SDF(df)