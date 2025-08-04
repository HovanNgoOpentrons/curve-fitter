import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, root_scalar

# === Configuration / Hyperparameters ===
TOTAL_HEIGHT = 14.95  # fixed overall stack height (mm)
OFFSET = 1  # downward bias on model-predicted height (mm)

# Soft user inputs and defaults
USER_SOFT_INPUTS = {
    # "R1": 1.25,
    "H1": 0.8
}
DEFAULTS = {"R1": 2, "H1": 0.3, "R2": 1.5, "H2": 12, "R3": 3.5}

SOFT_LOCK_REL_TOL = 0.8  # relative tolerance (larger = softer lock)
SOFT_PENALTY_SCALE = 100  # global scale for soft-lock penalty

# Endpoint enforcement weight
ENDPOINT_WEIGHT = 5000.0  # strong enforcement of last point

# Parameter names and bounds
ALL_NAMES = ["R1", "H1", "R2", "H2", "R3"]
BOUNDS_LOWER = [0.1, 0.05, 0.1, 0.1, 0.1]
BOUNDS_UPPER = [10, 5, 10, 20, 10]

# === Soft locks ===
def build_soft_locks(defaults, user_inputs, rel_tol=SOFT_LOCK_REL_TOL):
    soft = {}
    for name, target in user_inputs.items():
        base = defaults.get(name, target)
        tol = max(abs(base) * rel_tol, 1e-4)
        soft[name] = (target, tol)
    return soft

SOFT_LOCKS = build_soft_locks(DEFAULTS, USER_SOFT_INPUTS)

print(f"Optimizing parameters: {ALL_NAMES}")
print(f"Soft locks: {SOFT_LOCKS}")
print(f"Fixed total height: {TOTAL_HEIGHT}")

# === Pack/unpack ===
def unpack(free_vec):
    return list(free_vec)  # all five are free: [R1, H1, R2, H2, R3]

# === Data ===
data = np.array([
    [0, 0],
    [1, 0.1719166667],
    [2, 0.3438333333],
    [3, 0.51575],
    [4, 0.6876666667],
    [5, 0.8595833333],
    [6, 1.0315],
    [7, 1.203416667],
    [8, 1.375333333],
    [9, 1.54725],
    [10, 1.719166667],
    [11, 1.891083333],
    [12, 2.063],
    [13, 2.234916667],
    [14, 2.406833333],
    [15, 2.57875],
    [16, 2.72015],
    [17, 2.86155],
    [18, 3.00295],
    [19, 3.14435],
    [20, 3.28575],
    [21, 3.38695],
    [22, 3.48815],
    [23, 3.58935],
    [24, 3.69055],
    [25, 3.79175],
    [26, 3.86515],
    [27, 3.93855],
    [28, 4.01195],
    [29, 4.08535],
    [30, 4.15875],
    [31, 4.233574191],
    [32, 4.308398383],
    [33, 4.383222574],
    [34, 4.458046765],
    [35, 4.532870956],
    [36, 4.621380554],
    [37, 4.716707226],
    [38, 4.812033899],
    [39, 4.907360571],
    [40, 5.002687243],
    [41, 5.097345295],
    [42, 5.182496649],
    [43, 5.267648002],
    [44, 5.352799356],
    [45, 5.437950709],
    [46, 5.523102062],
    [47, 5.594340263],
    [48, 5.649501203],
    [49, 5.704662142],
    [50, 5.759823082],
    [51, 5.814984022],
    [52, 5.870144961],
    [53, 5.944199857],
    [54, 6.021276234],
    [55, 6.098352612],
    [56, 6.175428989],
    [57, 6.252505366],
    [58, 6.329581744],
    [59, 6.390464924],
    [60, 6.445216943],
    [61, 6.499968962],
    [62, 6.554720981],
    [63, 6.609473],
    [64, 6.664225019],
    [65, 6.729122236],
    [66, 6.801110405],
    [67, 6.873098573],
    [68, 6.945086742],
    [69, 7.017074911],
    [70, 7.089063079],
    [71, 7.161051248],
    [72, 7.221753176],
    [73, 7.281843809],
    [74, 7.341934442],
    [75, 7.402025075],
    [76, 7.462115708],
    [77, 7.522206341],
    [78, 7.575121991],
    [79, 7.611966285],
    [80, 7.648810578],
    [81, 7.685654871],
    [82, 7.722499165],
    [83, 7.759343458],
    [84, 7.796187752],
    [85, 7.838945612],
    [86, 7.893529718],
    [87, 7.948113824],
    [88, 8.00269793],
    [89, 8.057282037],
    [90, 8.111866143],
    [91, 8.166450249],
    [92, 8.221034355],
    [93, 8.274182582],
    [94, 8.325954326],
    [95, 8.37772607],
    [96, 8.429497814],
    [97, 8.481269558],
    [98, 8.533041302],
    [99, 8.584813046],
    [100, 8.636584791],
    [101, 8.686774066],
    [102, 8.736244907],
    [103, 8.785715748],
    [104, 8.835186588],
    [105, 8.884657429],
    [106, 8.93412827],
    [107, 8.983599111],
    [108, 9.033069952],
    [109, 9.09312525],
    [110, 9.154832882],
    [111, 9.216540513],
    [112, 9.278248145],
    [113, 9.339955777],
    [114, 9.401663408],
    [115, 9.46337104],
    [116, 9.525078672],
    [117, 9.56768719],
    [118, 9.597573043],
    [119, 9.627458896],
    [120, 9.657344749],
    [121, 9.687230602],
    [122, 9.717116455],
    [123, 9.747002308],
    [124, 9.776888161],
    [125, 9.810655947],
    [126, 9.852115495],
    [127, 9.893575043],
    [128, 9.935034591],
    [129, 9.976494139],
    [130, 10.01795369],
    [131, 10.05941323],
    [132, 10.10087278],
    [133, 10.14233233],
    [134, 10.18335776],
    [135, 10.20754239],
    [136, 10.23172702],
    [137, 10.25591166],
    [138, 10.28009629],
    [139, 10.30428092],
    [140, 10.32846555],
    [141, 10.35265019],
    [142, 10.37683482],
    [143, 10.40101945],
    [144, 10.42963573],
    [145, 10.47788314],
    [146, 10.52613056],
    [147, 10.57437797],
    [148, 10.62262539],
    [149, 10.6708728],
    [150, 10.71912022],
    [151, 10.76736763],
    [152, 10.81561505],
    [153, 10.86386246],
    [154, 10.91210988],
    [155, 10.9593571],
    [156, 10.9890962],
    [157, 11.01883529],
    [158, 11.04857439],
    [159, 11.07831348],
    [160, 11.10805258],
    [161, 11.13779167],
    [162, 11.16753077],
    [163, 11.19726987],
    [164, 11.22700896],
    [165, 11.25674806],
    [166, 11.28648715],
    [167, 11.33857883],
    [168, 11.3925114],
    [169, 11.44644396],
    [170, 11.50037652],
    [171, 11.55430908],
    [172, 11.60824164],
    [173, 11.6621742],
    [174, 11.71610677],
    [175, 11.77003933],
    [176, 11.82397189],
    [177, 11.87790445],
    [178, 11.93183701],
    [179, 11.97576438],
    [180, 12.01797136],
    [181, 12.06017835],
    [182, 12.10238534],
    [183, 12.14459232],
    [184, 12.18679931],
    [185, 12.22900629],
    [186, 12.27121328],
    [187, 12.31342027],
    [188, 12.35562725],
    [189, 12.39783424],
    [190, 12.44563828],
    [191, 12.49591029],
    [192, 12.54618231],
    [193, 12.59645432],
    [194, 12.64672634],
    [195, 12.69699835],
    [196, 12.74727037],
    [197, 12.79754238],
    [198, 12.8478144],
    [199, 12.89808642],
    [200, 12.94835843]
])
V_target = data[:, 0]
H_udv = data[:, 1]

# === Volume / height model ===
def volume_from_height(h, params):
    R1, H1, R2, H2, R3 = params
    H3 = TOTAL_HEIGHT - H1 - H2
    if any(p <= 0 for p in (R1, H1, R2, H2, R3)) or H3 <= 0:
        return np.inf

    if h <= H1:
        return np.pi * h**2 * (R1 - h / 3)

    V_cap = np.pi * H1**2 * (R1 - H1 / 3)

    if h <= H1 + H2:
        x = h - H1
        r_lower, r_upper = R2, R3
        r = r_lower + (r_upper - r_lower) * (x / H2)
        return V_cap + (np.pi * x / 3) * (r_lower**2 + r_lower * r + r**2)

    V_cone_full = (np.pi * H2 / 3) * (R2**2 + R2 * R3 + R3**2)

    if h <= TOTAL_HEIGHT:
        x = h - H1 - H2
        return V_cap + V_cone_full + np.pi * R3**2 * x

    return np.inf

def height_from_volume(V, params):
    if V <= 0:
        return 0.0

    def f(h):
        return volume_from_height(h, params) - V

    low, high = 0.0, TOTAL_HEIGHT
    f_low, f_high = f(low), f(high)

    if not (np.isfinite(f_low) and np.isfinite(f_high)):
        raise RuntimeError(f"Cannot bracket root: f_low={f_low}, f_high={f_high}, params={params}")

    if f_high < 0:
        high = TOTAL_HEIGHT * 1.5
        f_high = f(high)
        if f_low * f_high >= 0:
            raise RuntimeError(f"Cannot bracket root after extending high: f_low={f_low}, f_high={f_high}, params={params}")

    sol = root_scalar(f, bracket=[low, high], method="bisect", xtol=1e-6)
    if not sol.converged:
        raise RuntimeError(f"Root finding failed for V={V} with params={params}")
    return sol.root - OFFSET

# === Residual / cost function ===
def residuals(free_vec, V_data, H_data):
    params = unpack(free_vec)
    R1, H1, R2, H2, R3 = params
    H3 = TOTAL_HEIGHT - H1 - H2

    if H3 <= 0 or any(p <= 0 for p in (R1, H1, R2, H2, R3)):
        return np.ones(len(V_data) + 2) * 1e6  # invalid geometry

    try:
        model_heights = np.array([height_from_volume(V, params) for V in V_data])
    except RuntimeError:
        return np.ones(len(V_data) + 2) * 1e6  # root-finding failure

    # simple unweighted data residuals (no special beginning weight)
    height_residuals = model_heights - H_data

    # soft-lock penalty
    penalty = 0.0
    for i, name in enumerate(ALL_NAMES):
        if name in SOFT_LOCKS:
            target, tol = SOFT_LOCKS[name]
            penalty += ((params[i] - target) / tol) ** 2

    # enforce geometric consistency (H1 + H2 < TOTAL_HEIGHT)
    if H1 + H2 >= TOTAL_HEIGHT:
        penalty += 1e6 * (H1 + H2 - TOTAL_HEIGHT) ** 2

    penalty_scaled = penalty * SOFT_PENALTY_SCALE

    # endpoint enforcement (last data point)
    endpoint_res = model_heights[-1] - H_data[-1]
    endpoint_term = np.sqrt(ENDPOINT_WEIGHT) * endpoint_res

    return np.concatenate([height_residuals, [penalty_scaled, endpoint_term]])

# === Optimization driver ===
def generate_initial_guess():
    R1 = USER_SOFT_INPUTS.get("R1", DEFAULTS["R1"])
    H1 = USER_SOFT_INPUTS.get("H1", DEFAULTS["H1"])
    R2 = USER_SOFT_INPUTS.get("R2", DEFAULTS["R2"])
    H2 = USER_SOFT_INPUTS.get("H2", DEFAULTS["H2"])
    R3 = USER_SOFT_INPUTS.get("R3", DEFAULTS["R3"])
    if TOTAL_HEIGHT - H1 - H2 <= 0:
        scale = TOTAL_HEIGHT / (H1 + H2 + 1e-12)
        H1 *= 0.5 * scale
        H2 *= 0.5 * scale
    return np.array([R1, H1, R2, H2, R3])

def run_optimization_restarts(n_restarts=5, jitter_scale=0.1, random_seed=42):
    rng = np.random.default_rng(random_seed)
    best = None
    best_cost = np.inf
    base_x0 = generate_initial_guess()
    for i in range(n_restarts):
        perturb = 1 + rng.normal(0, jitter_scale, size=base_x0.shape)
        x0 = np.clip(base_x0 * perturb, BOUNDS_LOWER, BOUNDS_UPPER)
        result = least_squares(
            residuals,
            x0,
            bounds=(BOUNDS_LOWER, BOUNDS_UPPER),
            args=(V_target, H_udv),
            xtol=1e-10,
            ftol=1e-10,
            max_nfev=2000,
            verbose=2
        )
        if result.cost < best_cost:
            best_cost = result.cost
            best = result
        print(f"Restart {i+1}/{n_restarts}: cost={result.cost:.6g}")
    return best

# === Execute ===
best_result = run_optimization_restarts(n_restarts=10, jitter_scale=0.05, random_seed=123)

print("\nBest optimization result:")
for name, val in zip(ALL_NAMES, best_result.x):
    print(f"  {name} = {val:.5f}")

# === Visualization ===
fitted_params = unpack(best_result.x)
model_H = np.array([height_from_volume(V, fitted_params) for V in V_target])

plt.figure(figsize=(8, 5))
plt.plot(V_target, H_udv, 'ro', label="Measured data")
plt.plot(V_target, model_H, 'b-', label="Fitted model (with offset)")
plt.xlabel("Volume (uL)")
plt.ylabel("Height (mm)")
plt.title("Height vs Volume curve fit")
plt.legend()
plt.grid(True)
plt.show()
