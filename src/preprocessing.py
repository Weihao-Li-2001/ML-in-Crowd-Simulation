import pandas as pd
import numpy as np

from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

def preprocess_crowd_data(df, k=10):
    """
    Frame-based preprocessing: compute s_k and v from frame-to-frame differences.
    Preprocesses a pedestrian trajectory dataframe by:
    1. Converting units from cm to meters
    2. Computing average distance to k nearest neighbors
    3. Calculating relative positions to neighbors
    4. Computing velocities
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing pedestrian trajectory data with columns:
        ['FRAME', 'ID', 'X', 'Y', 'Z', ...]
    k : int, optional (default=10)
        Number of nearest neighbors to consider for neighborhood features
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe with additional features and cleaned data

    Notes:
    --------
    we handle the data by: (1) different units; (2) need to calculate s_k, x_r, y_r, v; (3) those with zero or very low speed should be neglected?
    
    Future Work:
    --------
    - better way to deal with not enough neighbors problem
    - data quality check (same id same frame with 2 v) (is frame even distributed)
    """
    
    # ==== Step 1: Convert units from centimeters to meters ==== 
    df = df.copy()
    # Convert position columns (X, Y, Z) from cm to m by dividing by 100
    df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']] / 100.0

    # ==== Step 2: Compute s_k (average distance to k nearest neighbors) and relative positions dx_i, dy_i ====

    # Initialize lists to store computed features
    s_k_list = []  # Will store average distance to k nearest neighbors for each point
    dx_all = []     # Will store relative x positions of neighbors
    dy_all = []     # Will store relative y positions of neighbors
    
    # Process each frame independently
    for frame, group in df.groupby('FRAME'):
        # Get coordinates of all pedestrians (including oneself) in current frame
        coords = group[['X', 'Y']].to_numpy()
        n = len(coords)  # Number of pedestrians in current frame

        # mark all people in this frame with 'np.nan' if there aren't enough persons(k+1)
        if n <= k+1:
            # Not enough neighbors → mark features as NaN
            s_k_list.extend([np.nan] * n)
            dx_all.extend([[np.nan] * k] * n)
            dy_all.extend([[np.nan] * k] * n)
            continue

        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(coords)
        # Query for k+1 neighbors because each point includes itself as first neighbor
        dists, idxs = tree.query(coords, k=k+1)

        # Process each pedestrian in current frame
        for i in range(n):
            neighbor_dists = dists[i][1:]  # Exclude self (first element)
            neighbor_idxs = idxs[i][1:]    # Exclude self

            # defensive programming in case some txt data has missing values
            if len(neighbor_dists) < k:
                s_k_list.append(np.nan)
                dx_all.append([np.nan] * k)
                dy_all.append([np.nan] * k)
            else:
                # Compute average distance to k nearest neighbors
                s_k_list.append(np.mean(neighbor_dists))
                
                # Compute relative positions (neighbor coords - current pedestrian coords)
                delta = coords[neighbor_idxs] - coords[i]
                dx_all.append(delta[:, 0].tolist())  # x differences
                dy_all.append(delta[:, 1].tolist())  # y differences

    # Add computed features to the DataFrame
    df['s_k'] = s_k_list  # Average distance to k nearest neighbors
    
    # Add relative position features (dx_1, dy_1, dx_2, dy_2, ..., dx_k, dy_k)
    for j in range(k):
        df[f'dx_{j+1}'] = [row[j] for row in dx_all]
        df[f'dy_{j+1}'] = [row[j] for row in dy_all]

    # Remove points with fewer than k neighbors (where s_k is NaN)
    df = df.dropna(subset=['s_k']).reset_index(drop=True)

    # ==== Step 3: Compute velocities ====
    # Sort by ID and FRAME to ensure chronological order for each pedestrian
    df = df.sort_values(by=['ID', 'FRAME']).reset_index(drop=True)
    
    # Get previous positions and frames for velocity calculation
    df['X_prev'] = df.groupby('ID')['X'].shift(1)
    df['Y_prev'] = df.groupby('ID')['Y'].shift(1)
    df['FRAME_prev'] = df.groupby('ID')['FRAME'].shift(1)

    # Compute differences between current and previous positions/frames
    dx = df['X'] - df['X_prev']
    dy = df['Y'] - df['Y_prev']
    dframe = df['FRAME'] - df['FRAME_prev']

    # Compute velocity (distance / time) and convert to m/s
    # 16 frames per second
    frame_rate = 16
    dist = np.sqrt(dx**2 + dy**2)
    df['v'] = (dist / dframe) * frame_rate  # Velocity in m/s

    # Forward fill NaN velocities (first frame of each pedestrian)
    df['v'] = df.groupby('ID')['v'].transform(lambda x: x.bfill())
    
    # ==== Step 4: Apply 1-second rolling average ====
    window = 2  # 1s = 16 frames

    # For each pedestrian (ID), take sliding average of v and s_k
    df['v_smooth'] = (
        df.groupby('ID')['v']
          .transform(lambda x: x.rolling(window=window, min_periods=1, center=True).mean()) # 
    )
    
    df['s_k_smooth'] = (
        df.groupby('ID')['s_k']
          .transform(lambda x: x.rolling(window=window, min_periods=1, center=True).mean())
    )

    # Clean up temporary columns used for velocity calculation
    df.drop(columns=['X_prev', 'Y_prev', 'FRAME_prev'], inplace=True)
    
    df = df.reset_index(drop=True)

    # Method 3: Filter method (Useless)
    # df = df[df['v'] >= 0.05].reset_index(drop=True)
    
    return df

