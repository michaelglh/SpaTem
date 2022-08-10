# visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

def bound3d(pts, low, high):
    """Check it pts line in bound of a box [low, high)

    Args:
        pts (array): array of points
        low (float): lower boundary
        high (float): higher boundary

    Returns:
        array: boolean of points within boundary
    """    
    return (pts[0]<high) & (pts[0]>=low) & (pts[1]<high) & (pts[1]>=low) & (pts[2]<high) & (pts[2]>=low)

def seq3d(seqs, T, W, K, fname):
    """Visualize the 3d sequences

    Args:
        seqs (list): list of sequences
        T (int): number of time steps
        W (int): network size
        K (int): number of sequneces
        fname (string): figure path

    Returns:
        None: None
    """    
    cm = plt.get_cmap('jet', K) 
    cs = [cm(k) for k in range(K)]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    def update_img(t, seqs, ax):
        plt.cla()
        for k, c in zip(range(K), cs):
            if seqs[k][t].shape[0] > 0:
                xs, ys, zs = seqs[k][t][0], seqs[k][t][1], seqs[k][t][2]
                idx = bound3d([xs, ys, zs], -1, W+1)
                im = ax.scatter(xs[idx], ys[idx], zs[idx], s=1, color=c)
        ax.set_xlim([-1,W+1])
        ax.set_ylim([-1,W+1])
        ax.set_zlim([-1,W+1])
        return im

    ani = animation.FuncAnimation(fig, update_img, T, fargs=(seqs,ax), interval=100, blit=False)
    ani.save(fname, writer='pillow')

def displane(p, A, Q):
    """Distance of a point p to a plane (A, Q)

    Args:
        p (array): point position
        A (array): direction of plane, assuming normalized
        Q (array): origin of plane

    Returns:
        _type_: _description_
    """    
    return np.absolute(np.dot(p-Q, A))

def exp2d(seqs, K, T, A, Q, h=1):
    """Observation of 3d sequences on a given 2d plane

    Args:
        seqs (list): list of sequences
        K (int): number of sequences
        T (int): number of time steps
        A (array): direction of plane, assuming normalized
        Q (array): origin of plane
        h (float, optional): width of the observation plane sheet. Defaults to 1.

    Returns:
        array: observed sequences, projection of them, basis on the plane used for projection
    """    
    # basis on the plane
    ex = np.ones(3)
    if np.linalg.norm(np.cross(ex, A)) == 0:
        print("Oops, careful with the projection...")
        ex = np.array([1., 0., 0.])
    ex -= ex.dot(A) * A
    ey = np.cross(A, ex)
    ex = ex/np.linalg.norm(ex)
    ey = ey/np.linalg.norm(ey)

    # observation and projection
    seq_2d = [[] for _ in range(K)]
    prj_2d = [[] for _ in range(K)]
    for k in range(K):
        for t in range(T):
            cluster_seq = []
            cluster_prj = []
            for p in seqs[k][t].T:
                if displane(p, A, Q) < h:
                    cluster_seq.append(p)
                    cluster_prj.append([np.dot(ex, p-Q), np.dot(ey, p-Q)])
            if len(cluster_seq) == 0:
                cluster_seq = np.array([]).reshape(0,3)
            if len(cluster_prj) == 0:
                cluster_prj = np.array([]).reshape(0,2)
            seq_2d[k].append(np.array(cluster_seq, dtype=int).T)
            prj_2d[k].append(np.array(cluster_prj).T)

    return seq_2d, prj_2d, ex, ey

def seq2d(seqs, T, W, K, A, Q, fname, h=1):
    """Visualization of 2d observation

    Args:
        seqs (list): list of sequences
        K (int): number of sequences
        T (int): number of time steps
        W (int): network size
        A (array): direction of plane, assuming normalized
        Q (array): origin of plane
        fname (string): figure path
        h (float, optional): width of the observation plane sheet. Defaults to 1.

    Returns:
        array: basis of projection on to plane (A, Q)
    """    
    # 2d observation
    seq_2d, prj_2d, ex, ey = exp2d(seqs, K, T, A, Q, h)
    # observation plane
    if A.dot(np.array([0,0,1])) != 0:
        xx, yy = np.meshgrid(range(W), range(W))
        zz = (np.dot(Q, A) - A[0]*xx - A[1]*yy)/A[2]
        zz = np.minimum(np.maximum(zz, 0), W)
    elif A.dot(np.array([0,1,0])) != 0:
        xx, zz = np.meshgrid(range(W), range(W))
        yy = (np.dot(Q, A) - A[0]*xx - A[2]*zz)/A[1]
        yy = np.minimum(np.maximum(yy, 0), W)
    elif A.dot(np.array([1,0,0])) != 0:
        yy, zz = np.meshgrid(range(W), range(W))
        xx = (np.dot(Q, A) - A[1]*yy - A[2]*zz)/A[0]
        xx = np.minimum(np.maximum(xx, 0), W)
    else:
        print('Oops, something is biting!!!')
    # projection range
    gx, gy, gz = np.meshgrid(np.linspace(-1, W+1, 2), np.linspace(-1, W+1, 2), np.linspace(-1, W+1, 2))
    grids = np.stack([gx, gy, gz]).reshape(3,-1)
    Qv = Q.reshape(3,1)
    lowx, highx = np.min(ex.reshape(1,3)@(grids-Qv)), np.max(ex.reshape(1,3)@(grids-Qv))
    lowy, highy = np.min(ey.reshape(1,3)@(grids-Qv)), np.max(ey.reshape(1,3)@(grids-Qv))
    # color setting
    cm = plt.get_cmap('jet', K) 
    cs = [cm(k) for k in range(K)]

    def update_3d(t, data, ax):
        plt.cla()
        if t %2 == 0:
            im = ax.scatter(0., 0., s=1, alpha=0.1)    # force update
        else:
            im = ax.scatter(1., 1., s=1, alpha=0.1)    # force update
        # 2d plane in 3d visualization
        im = ax.plot_surface(xx, yy, zz, alpha=0.5)
        for k, c in zip(range(K), cs):
            if data[k][t].shape[0] > 0:
                xs, ys, zs = data[k][t][0], data[k][t][1], data[k][t][2]
                idx = bound3d([xs, ys, zs], -1, W+1)
                im = ax.scatter(xs[idx], ys[idx], zs[idx], s=1, color=c)
        ax.set_xlim([-1,W+1])
        ax.set_ylim([-1,W+1])
        ax.set_zlim([-1,W+1])
        return im

    def update_2d(t, data, ax):
        plt.cla()
        if t %2 == 0:
            im = ax.scatter(0., 0., s=1, alpha=0.1)    # force update
        else:
            im = ax.scatter(1., 1., s=1, alpha=0.1)    # force update
        # 2d projection
        for k, c in zip(range(K), cs):
            if data[k][t].shape[0] > 0:
                im = ax.scatter(data[k][t][0], data[k][t][1], s=1, color=c)
        ax.set_xlim([lowx,highx])
        ax.set_ylim([lowy,highy])
        return im

    # plane observation in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ani = animation.FuncAnimation(fig, update_3d, T, fargs=(seq_2d, ax), interval=100, blit=False)
    ani.save(fname + '_obs.gif', writer='pillow')
    plt.close()

    # plane observation of projection
    fig = plt.figure()
    ax = plt.axes()
    ani = animation.FuncAnimation(fig, update_2d, T, fargs=(prj_2d, ax), interval=100, blit=False)
    ani.save(fname + '_prj.gif', writer='pillow')
    plt.close()

    return ex, ey
