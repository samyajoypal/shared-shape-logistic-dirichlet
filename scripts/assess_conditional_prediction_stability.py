"""Read-only stability of the conditional mean of the same log target.

No fitting, component matching, held-out selection or population-MSE claim.
Every original covariate row and every declared bootstrap replication is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_solve
from scipy.special import digamma, gammaln, logsumexp, polygamma

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("shared_ld", "full_ld", "gaussian_shared_correlation")
PLAN = {
    "scope": "All three original primary application representations, all original rows, all199 declared bootstrap replicates",
    "target": "Conditional mean of the analysis-coordinate log target, given all other analysis coordinates",
    "targets": {"olive3_oleic": "linoleic/oleic", "olive6_oleic": "linoleic/oleic", "crabs_log5": "BD"},
    "methods": list(METHODS), "bootstrap_repetitions": 199,
    "grid_weighting": "Equal weight for every original observation; bootstrap does not change the evaluation grid",
    "main_estimate": "Mean over grid rows of sample variance over all199 bootstrap predictions",
    "eligibility": "Use every finite saved fit, retaining boundary and nonstationary flags; no replacement or selection by score",
    "missing_policy": "Primary conclusions require all199 draws; conditional available-draw summaries retain explicit denominators",
    "uncertainty": "Paired delete-one-bootstrap jackknife Monte Carlo uncertainty; covariate rows are not independent replications",
    "interpretation": "Conditional resampling stability of bounded numerical fitting procedures, not MSE, coverage, equivalence or proven population efficiency",
    "refitting": False,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def conditioning_values(y, target, d):
    y = np.asarray(y, float)
    if y.ndim != 2 or y.shape[1] != d or not 0 <= target < d:
        raise ValueError("Invalid conditioning grid or target")
    keep = np.array([m for m in range(d) if m != target], dtype=int)
    if not np.isfinite(y[:, keep]).all():
        raise ValueError("Conditioning coordinates must be finite")
    return y, keep


def ld_conditional_mean(theta, y, k, d, target, shared_shapes, gradient=False):
    """Exact conditional mean and optional derivative in raw eta,alpha,a,b.

    The target column of y is never used.  The derivative treats the
    conditioning grid and its fixed training transformation as constants.
    """
    y, keep = conditioning_values(y, target, d)
    theta = np.asarray(theta, float)
    p, n, shape_count = d + 1, len(y), (d + 1 if shared_shapes else k * (d + 1))
    if theta.shape != (k - 1 + shape_count + 2 * k * d,) or not np.isfinite(theta).all():
        raise ValueError("Invalid LD parameter vector")
    offset = k - 1
    logits = np.r_[theta[:offset], 0.]
    logpi = logits - logsumexp(logits)
    alpha = theta[offset:offset + shape_count].reshape(-1, p)
    if shared_shapes:
        alpha = np.broadcast_to(alpha, (k, p))
    offset += shape_count
    a = theta[offset:offset + k*d].reshape(k, d)
    b = theta[offset + k*d:].reshape(k, d)
    if np.any(alpha <= 0) or np.any(a <= 0):
        raise ValueError("LD shapes and inverse scales must be positive")
    observed_alpha = np.r_[keep, d]
    other_sum = alpha[:, observed_alpha].sum(axis=1)
    z = y[:, None, keep] * a[None, :, keep] + b[None, :, keep]
    logc = logsumexp(np.concatenate([np.zeros((n, k, 1)), z], axis=2), axis=2)
    q = np.exp(z - logc[:, :, None])
    log_marginal = (gammaln(other_sum) - gammaln(alpha[:, observed_alpha]).sum(axis=1)
                    + np.log(a[:, keep]).sum(axis=1))[None, :]
    log_marginal = log_marginal + (z * alpha[None, :, keep]).sum(axis=2) - other_sum[None, :] * logc
    terms = logpi[None, :] + log_marginal
    posterior = np.exp(terms - logsumexp(terms, axis=1)[:, None])
    component_mean = (logc + digamma(alpha[:, target])[None, :]
                      - digamma(other_sum)[None, :] - b[:, target][None, :]) / a[:, target][None, :]
    prediction = (posterior * component_mean).sum(axis=1)
    if not gradient:
        return prediction
    residual_weight = posterior * (component_mean - prediction[:, None])
    shape_marginal = np.zeros((n, k, p))
    shape_marginal[:, :, keep] = (digamma(other_sum)[None, :, None]
        - digamma(alpha[:, keep])[None, :, :] + z - logc[:, :, None])
    shape_marginal[:, :, d] = digamma(other_sum)[None, :] - digamma(alpha[:, d])[None, :] - logc
    shape_mean = np.broadcast_to((-polygamma(1, other_sum)/a[:, target])[None, :, None], (n, k, p)).copy()
    shape_mean[:, :, target] = (polygamma(1, alpha[:, target])/a[:, target])[None, :]
    shape_derivative = posterior[:, :, None]*shape_mean + residual_weight[:, :, None]*shape_marginal
    a_marginal = np.zeros((n, k, d)); b_marginal = np.zeros_like(a_marginal)
    score_z = alpha[None, :, keep] - other_sum[None, :, None]*q
    a_marginal[:, :, keep] = 1/a[None, :, keep] + y[:, None, keep]*score_z
    b_marginal[:, :, keep] = score_z
    a_mean = np.zeros_like(a_marginal); b_mean = np.zeros_like(a_marginal)
    a_mean[:, :, target] = -component_mean/a[:, target][None, :]
    b_mean[:, :, target] = -1/a[:, target][None, :]
    a_mean[:, :, keep] = q*y[:, None, keep]/a[:, target][None, :, None]
    b_mean[:, :, keep] = q/a[:, target][None, :, None]
    a_derivative = posterior[:, :, None]*a_mean + residual_weight[:, :, None]*a_marginal
    b_derivative = posterior[:, :, None]*b_mean + residual_weight[:, :, None]*b_marginal
    derivative = np.concatenate([residual_weight[:, :k-1],
        shape_derivative.sum(axis=1) if shared_shapes else shape_derivative.reshape(n, -1),
        a_derivative.reshape(n, -1), b_derivative.reshape(n, -1)], axis=1)
    return prediction, derivative


def gaussian_conditional_mean(theta, y, k, d, target, gradient=False):
    """Exact common-correlation Gaussian conditional mean in native fit coordinates."""
    y, keep = conditioning_values(y, target, d)
    c = 0 if d == 1 else 1 if d == 2 else d
    theta = np.asarray(theta, float)
    if theta.shape != (k-1+2*k*d+c,) or not np.isfinite(theta).all():
        raise ValueError("Invalid Gaussian parameter vector")
    logits = np.r_[theta[:k-1], 0.]; logpi = logits-logsumexp(logits)
    means = theta[k-1:k-1+k*d].reshape(k, d)
    std = np.exp(theta[k-1+k*d:k-1+2*k*d].reshape(k, d))
    correlation = theta[-c:] if c else np.empty(0)
    if not np.isfinite(std).all() or np.any(std <= 0) or np.any(correlation < 0) or np.any(correlation >= 1):
        raise ValueError("Invalid Gaussian scale or correlation")
    R = np.eye(d)
    if d == 2:
        R[0, 1] = R[1, 0] = correlation[0]
    elif d >= 3:
        R = np.outer(correlation, correlation); np.fill_diagonal(R, 1.)
    n = len(y)
    z = (y[:, None, keep] - means[None, :, keep])/std[None, :, keep]
    if len(keep):
        lower = np.linalg.cholesky(R[np.ix_(keep, keep)])
        inverse = cho_solve((lower, True), np.eye(len(keep)))
        coefficient = inverse @ R[keep, target]
        precision_residual = z @ inverse
        log_marginal = (-.5*len(keep)*np.log(2*np.pi)-np.log(np.diag(lower)).sum()
            - np.log(std[:, keep]).sum(axis=1)[None, :] - .5*(z*precision_residual).sum(axis=2))
        correction = std[:, target][None, :]*(z@coefficient)
    else:
        inverse = np.empty((0, 0)); coefficient = np.empty(0)
        precision_residual = z; log_marginal = np.zeros((n, k)); correction = np.zeros((n, k))
    component_mean = means[:, target][None, :] + correction
    terms = logpi[None, :] + log_marginal
    posterior = np.exp(terms-logsumexp(terms, axis=1)[:, None])
    prediction = (posterior*component_mean).sum(axis=1)
    if not gradient:
        return prediction
    residual_weight = posterior*(component_mean-prediction[:, None])
    mean_gradient = np.zeros((n, k, d)); scale_gradient = np.zeros_like(mean_gradient)
    mean_gradient[:, :, target] = posterior
    scale_gradient[:, :, target] = posterior*correction
    mean_gradient[:, :, keep] = (-posterior[:, :, None]*std[:, target][None, :, None]*coefficient[None, None, :]
        + residual_weight[:, :, None]*precision_residual)/std[None, :, keep]
    scale_gradient[:, :, keep] = (-posterior[:, :, None]*std[:, target][None, :, None]*z*coefficient[None, None, :]
        + residual_weight[:, :, None]*(z*precision_residual-1))
    correlation_gradient = np.zeros((n, c))
    for j in range(c):
        dR = np.zeros((d, d))
        if d == 2:
            dR[0, 1] = dR[1, 0] = 1.
        else:
            dR[j, :] = correlation; dR[:, j] += correlation; dR[j, j] = 0.
        sub = dR[np.ix_(keep, keep)]
        dc = inverse @ (dR[keep, target] - sub@coefficient)
        mean_derivative = std[:, target][None, :]*(z@dc)
        marginal_derivative = .5*((precision_residual@sub)*precision_residual).sum(axis=2)-.5*np.trace(inverse@sub)
        correlation_gradient[:, j] = (posterior*mean_derivative+residual_weight*marginal_derivative).sum(axis=1)
    derivative = np.concatenate([residual_weight[:, :k-1], mean_gradient.reshape(n, -1),
                                  scale_gradient.reshape(n, -1), correlation_gradient], axis=1)
    return prediction, derivative


def predict(params, method, y, target, center, scale, gradient=False):
    """Return conditional mean on the original analysis log scale."""
    y, center, scale = np.asarray(y, float), np.asarray(center, float), np.asarray(scale, float)
    d, k = int(params["d"]), int(params["k"])
    if center.shape != (d,) or scale.shape != (d,) or np.any(scale <= 0):
        raise ValueError("Invalid saved training transformation")
    transformed = (y-center)/scale
    if method in ("shared_ld", "full_ld"):
        result = ld_conditional_mean(params["theta"], transformed, k, d, target, method == "shared_ld", gradient)
    elif method == "gaussian_shared_correlation":
        result = gaussian_conditional_mean(params["theta"], transformed, k, d, target, gradient)
    else:
        raise ValueError("Unsupported method")
    if gradient:
        return center[target]+scale[target]*result[0], scale[target]*result[1]
    return center[target]+scale[target]*result


def variance_and_delete_one(predictions):
    """Average pointwise sample variance and its paired jackknife replicates."""
    predictions = np.asarray(predictions, float)
    if predictions.ndim != 2 or not np.isfinite(predictions).all() or predictions.shape[0] < 2 or not predictions.shape[1]:
        raise ValueError("Need at least two finite prediction curves on a nonempty fixed grid")
    B = len(predictions)
    centered = predictions-predictions.mean(axis=0)
    squared_norm = np.mean(centered**2, axis=1)
    variance = float(squared_norm.sum()/(B-1))
    leave_one = ((B-1)*variance-B/(B-1)*squared_norm)/(B-2) if B > 2 else np.full(B, np.nan)
    if np.any(leave_one < -1e-12*(1+variance)):
        raise ArithmeticError("Negative delete-one variance")
    return variance, np.maximum(leave_one, 0.)


def jackknife_mcse(values):
    values = np.asarray(values, float)
    if len(values) < 3 or not np.isfinite(values).all():
        return np.nan
    return float(np.sqrt((len(values)-1)/len(values)*np.sum((values-values.mean())**2)))


def summarize_method(predictions, full_prediction, requested):
    predictions, full_prediction = np.asarray(predictions), np.asarray(full_prediction)
    valid = np.isfinite(predictions).all(axis=1)
    draws = predictions[valid]
    result = {"bootstrap_requested": requested, "bootstrap_available": int(valid.sum()),
              "bootstrap_unavailable": int(requested-valid.sum()), "all_draws_available": bool(valid.sum() == requested),
              "grid_rows": predictions.shape[1], "variance": None, "variance_mcse": None,
              "mean_squared_bootstrap_shift": None, "mean_squared_deviation_from_full": None}
    if len(draws) >= 2:
        variance, leave_one = variance_and_delete_one(draws)
        result.update(variance=variance, variance_mcse=jackknife_mcse(leave_one))
        if np.isfinite(full_prediction).all():
            shift = float(np.mean((draws.mean(axis=0)-full_prediction)**2))
            deviation = float(np.mean((draws-full_prediction[None, :])**2))
            np.testing.assert_allclose(deviation, (len(draws)-1)/len(draws)*variance+shift, rtol=1e-11, atol=1e-13)
            result.update(mean_squared_bootstrap_shift=shift, mean_squared_deviation_from_full=deviation)
    return result


def summarize_pair(left, right, left_full, right_full, requested):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("Paired bootstrap predictions must have identical replicate and grid shapes")
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    L, R = left[valid], right[valid]
    row = {"bootstrap_requested": requested, "both_available": int(valid.sum()),
           "all_draws_available": bool(valid.sum() == requested),
           "variance_left": None, "variance_right": None, "variance_difference": None,
           "variance_difference_mcse": None, "variance_ratio": None, "log_variance_ratio_mcse": None,
           "ratio_mc95_low": None, "ratio_mc95_high": None, "variance_reduction_percent": None,
           "full_prediction_mean_difference": None, "full_prediction_rms_difference": None}
    if len(L) >= 2:
        vl, jl = variance_and_delete_one(L); vr, jr = variance_and_delete_one(R)
        row.update(variance_left=vl, variance_right=vr, variance_difference=vl-vr,
                   variance_difference_mcse=jackknife_mcse(jl-jr))
        if vl > 0 and vr > 0:
            ratio = vl/vr
            se = jackknife_mcse(np.log(jl)-np.log(jr)) if np.all(jl > 0) and np.all(jr > 0) else np.nan
            row.update(variance_ratio=ratio, log_variance_ratio_mcse=se,
                       ratio_mc95_low=np.exp(np.log(ratio)-1.96*se),
                       ratio_mc95_high=np.exp(np.log(ratio)+1.96*se),
                       variance_reduction_percent=100*(1-ratio))
    if np.isfinite(left_full).all() and np.isfinite(right_full).all():
        delta = np.asarray(left_full)-np.asarray(right_full)
        row.update(full_prediction_mean_difference=float(delta.mean()),
                   full_prediction_rms_difference=float(np.sqrt(np.mean(delta**2))))
    return row


class IntegrityError(RuntimeError):
    pass


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class SavedInputs:
    """Bind parameter points to the same original tasks and transformations."""

    def __init__(self, primary, gaussian, refined=None):
        self.primary, self.gaussian = Path(primary).resolve(), Path(gaussian).resolve()
        self.refined = Path(refined).resolve() if refined else None
        self.hashes, self.cache = {}, {}
        self.spec = self.json(self.primary/"specification.json")
        self.primary_hash = self.hashes[str(self.primary/"specification.json")]
        self.gspec = self.json(self.gaussian/"manifest.json")
        self.gaussian_hash = self.hashes[str(self.gaussian/"manifest.json")]
        if self.gspec["source_kind"] != "applications" or self.gspec["primary_metadata_sha256"] != self.primary_hash:
            raise IntegrityError("Gaussian run belongs to a different primary application specification")
        for relative, expected in self.spec.get("source_hashes", {}).items():
            self.bytes(self.primary/"executed_sources"/relative, expected)
        for relative, expected in self.gspec.get("source_sha256", {}).items():
            self.bytes(self.gaussian/"source_snapshot"/relative, expected)
        if any(t["task_index"] != i for i, t in enumerate(self.spec["tasks"])):
            raise IntegrityError("Noncanonical primary task map")
        if any(t["task_index"] != i for i, t in enumerate(self.gspec["tasks"])):
            raise IntegrityError("Noncanonical Gaussian task map")
        self.gaussian_tasks = {t["primary_task_index"]: t for t in self.gspec["tasks"]}
        if len(self.gaussian_tasks) != len(self.gspec["tasks"]):
            raise IntegrityError("Duplicate Gaussian primary task mapping")
        self.rspec, self.refined_tasks = None, {}
        if self.refined:
            self.rspec = self.json(self.refined/"manifest.json")
            self.refined_hash = self.hashes[str(self.refined/"manifest.json")]
            if (self.rspec["metadata_sha256"]["primary"] != self.primary_hash
                    or self.rspec["metadata_sha256"]["gaussian"] != self.gaussian_hash):
                raise IntegrityError("Refinement belongs to different original application inputs")
            if any(t["task_index"] != i for i, t in enumerate(self.rspec["tasks"])):
                raise IntegrityError("Noncanonical refinement task map")
            self.refined_tasks = {t["primary_task_index"]:t for t in self.rspec["tasks"]}
            if len(self.refined_tasks) != len(self.rspec["tasks"]):
                raise IntegrityError("Duplicate refined primary task mapping")
            for relative, expected in self.rspec["sources_sha256"].items():
                self.bytes(self.refined/"source_snapshot"/relative, expected)

    def bytes(self, path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        value = hashlib.sha256(raw).hexdigest()
        if expected is not None and value != expected:
            raise IntegrityError("Input hash mismatch: "+str(path))
        if str(path) in self.hashes and self.hashes[str(path)] != value:
            raise IntegrityError("Input changed during analysis: "+str(path))
        self.hashes[str(path)] = value
        return raw

    def json(self, path):
        path = Path(path).resolve()
        if str(path) not in self.cache:
            self.cache[str(path)] = json.loads(self.bytes(path))
        return self.cache[str(path)]

    def variants(self):
        values = [v for v in self.spec["config"]["variants"] if v.get("primary")]
        if {v["id"] for v in values} != set(PLAN["targets"]):
            raise IntegrityError("Primary representations differ from the declared analysis plan")
        if self.spec["config"]["bootstrap_repetitions"] != PLAN["bootstrap_repetitions"]:
            raise IntegrityError("Bootstrap count differs from the frozen analysis plan")
        return values

    def grid(self, variant):
        relative = "data/"+("oliveoil.csv" if variant["dataset"] == "oliveoil" else "crabs.csv")
        raw = self.bytes(self.primary/"executed_sources"/relative, self.spec["data_hashes"][relative])
        frame = pd.read_csv(io.BytesIO(raw))
        values = frame[variant["columns"]].to_numpy(float)+variant.get("add_to_all", 0.)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise IntegrityError("Original scientific measurements are not positive and finite")
        if variant["dataset"] == "oliveoil":
            reference = variant["columns"].index(variant["reference"])
            keep = [m for m in range(values.shape[1]) if m != reference]
            y = np.log(values[:, keep]/values[:, reference, None])
            names = [variant["columns"][m]+"/"+variant["reference"] for m in keep]
        elif variant["transform"] == "log":
            y, names = np.log(values), variant["columns"]
        else:
            raise IntegrityError("This protocol is limited to the original log targets")
        target_name = PLAN["targets"][variant["id"]]
        if variant["conditional_target"] != target_name:
            raise IntegrityError("Target changed from the original prediction question")
        return y, names.index(target_name), names

    def tasks(self, variant):
        tasks = [t for t in self.spec["tasks"] if t["variant"] == variant["id"]
                 and t["bounds"] == "primary" and t["kind"] in ("full", "bootstrap")]
        full = [t for t in tasks if t["kind"] == "full"]
        bootstrap = sorted((t for t in tasks if t["kind"] == "bootstrap"), key=lambda t:t["repeat"])
        if len(full) != 1 or [t["repeat"] for t in bootstrap] != list(range(PLAN["bootstrap_repetitions"])):
            raise IntegrityError("Missing or duplicate declared full/bootstrap task")
        return full[0], bootstrap

    def transform(self, task, y):
        directory = self.primary/"tasks"/f"{task['task_index']:05d}"
        split = self.json(directory/"split.json")
        completed = self.json(directory/"completion.json")
        if (completed["task"] != task or completed["specification_sha256"] != self.primary_hash
                or any(split.get(key) != value for key, value in task.items())):
            raise IntegrityError("Original task, split or completion identity mismatch")
        center, scale = np.asarray(split["center"]), np.asarray(split["scale"])
        train = y[np.asarray(task["train"], dtype=int)]
        if (not np.isfinite(scale).all() or np.any(scale <= 0)
                or not np.allclose(center, train.mean(axis=0), rtol=1e-12, atol=1e-12)
                or not np.allclose(scale, train.std(axis=0), rtol=1e-12, atol=1e-12)):
            raise IntegrityError("Saved transform differs from the original bootstrap training rows")
        return center, scale, completed

    def parameters(self, task, method, y):
        center, scale, completed = self.transform(task, y)
        if self.refined:
            mapped = self.refined_tasks[task["task_index"]]
            directory = self.refined/"tasks"/f"task_{mapped['task_index']:06d}"
            complete = self.json(directory/"completion.json")
            value = self.json(directory/(method+".json"))
            binding = self.json(directory/"input_binding.json")
            identity = {"manifest_sha256":self.refined_hash, "task":mapped,
                        "input_binding_sha256":semantic_hash(binding)}
            if (complete.get("status") != "complete"
                    or any(complete.get(key) != val or value.get(key) != val for key, val in identity.items())):
                raise IntegrityError("Refined fit/completion/binding identity mismatch")
            if binding.get("task") != mapped or value.get("method") != method:
                raise IntegrityError("Refined method or bound task mismatch")
            roots = {"primary":self.primary, "gaussian":self.gaussian}
            for item in binding["files"]:
                if item["source"] not in roots:
                    raise IntegrityError("Unexpected input source for full/bootstrap refinement")
                self.bytes(roots[item["source"]]/item["relative_path"], item["sha256"])
            transform = value["training_transform"]
            if not np.array_equal(transform["center"], center) or not np.array_equal(transform["scale"], scale):
                raise IntegrityError("Refinement changed the training transformation")
            record, params = value["row"], value.get("parameters")
            listed = [r for r in complete["rows"] if r["method"] == method]
            if len(listed) != 1 or listed[0] != record:
                raise IntegrityError("Refined result row differs from task completion")
        elif method in ("shared_ld", "full_ld"):
            value = self.json(self.primary/"tasks"/f"{task['task_index']:05d}"/(method+".json"))
            record, params = value["record"], value.get("parameters")
            if value["specification_sha256"] != self.primary_hash:
                raise IntegrityError("LD parameter checkpoint belongs to another specification")
            listed = [r for r in completed["records"] if r["method"] == method]
            if len(listed) != 1 or listed[0] != record:
                raise IntegrityError("LD checkpoint record differs from completed task")
        else:
            mapped = self.gaussian_tasks[task["task_index"]]
            directory = self.gaussian/"tasks"/f"task_{mapped['task_index']:06d}"
            value = self.json(directory/"result.json")
            binding = self.json(directory/"input_binding.json")
            if (value["status"] != "complete" or value["task"] != mapped
                    or value["manifest_sha256"] != self.gaussian_hash
                    or semantic_hash(binding) != value["input_binding_sha256"]
                    or binding["primary_task_index"] != task["task_index"]):
                raise IntegrityError("Gaussian parameter/task/binding identity mismatch")
            split_name = f"tasks/{task['task_index']:05d}/split.json"
            if split_name not in binding["input_sha256"]:
                raise IntegrityError("Gaussian input binding omits the training transformation")
            for relative, expected in binding["input_sha256"].items():
                self.bytes(self.primary/relative, expected)
            record, params = value["row"], value.get("parameters")
        expected_task_index = mapped["task_index"] if self.refined or method.startswith("gaussian") else task["task_index"]
        if record["method"] != method or record["variant"] != task["variant"] or record["task_index"] != expected_task_index:
            raise IntegrityError("Selected parameter record has a different method/task identity")
        if params is None:
            raise ValueError("No finite saved parameter point")
        if params["method"] != method or int(params["d"]) != y.shape[1]:
            raise IntegrityError("Saved parameter family or dimension mismatch")
        return params, center, scale, record


def run_analysis(primary, gaussian, output, protocol, refined=None, cv_summary=None):
    output = Path(output).resolve()
    for path in (primary, gaussian, refined):
        if path and (output == Path(path).resolve() or Path(path).resolve() in output.parents):
            raise ValueError("Analysis output must be separate from every frozen input run")
    inputs = SavedInputs(primary, gaussian, refined)
    output.mkdir(parents=True, exist_ok=False)
    protocol = Path(protocol)
    manifest = {"created_utc":datetime.now(timezone.utc).isoformat(), "plan":PLAN,
                "procedure":"refined" if refined else "original", "refitting":False,
                "protocol_sha256":digest(protocol), "analysis_source_sha256":digest(__file__),
                "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__},
                "input_roots":{"primary":str(inputs.primary),"gaussian":str(inputs.gaussian),"refined":str(inputs.refined) if refined else None}}
    write_json(output/"analysis_plan.json", manifest)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (output/protocol.name).write_bytes(protocol.read_bytes())
    diagnostics, summaries, comparisons, grids = [], [], [], []
    for variant in inputs.variants():
        y, target, names = inputs.grid(variant)
        full_task, bootstrap_tasks = inputs.tasks(variant)
        B, n = len(bootstrap_tasks), len(y)
        predictions = {m:np.full((B, n), np.nan) for m in METHODS}
        full = {m:np.full(n, np.nan) for m in METHODS}
        flags = {m:{"score":np.zeros(B, bool),"interior":np.zeros(B, bool)} for m in METHODS}
        for task in [full_task]+bootstrap_tasks:
            for method in METHODS:
                diagnostic = {"variant":variant["id"],"method":method,"kind":task["kind"],
                    "bootstrap_index":task["repeat"] if task["kind"] == "bootstrap" else None,
                    "primary_task_index":task["task_index"],"prediction_available":False,
                    "source_failed":None,"score_converged":None,"boundary_hit":None,"reason":None}
                try:
                    params, center, scale, record = inputs.parameters(task, method, y)
                    values = predict(params, method, y, target, center, scale)
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Nonfinite conditional mean on the original grid")
                    if task["kind"] == "full":
                        full[method] = values
                    else:
                        j = task["repeat"]; predictions[method][j] = values
                        flags[method]["score"][j] = record.get("score_converged") is True
                        flags[method]["interior"][j] = record.get("boundary_hit") is False
                    diagnostic.update(prediction_available=True,source_failed=record.get("failed"),
                        score_converged=record.get("score_converged"),boundary_hit=record.get("boundary_hit"))
                except IntegrityError:
                    raise
                except (FileNotFoundError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    diagnostic["reason"] = type(exc).__name__+": "+str(exc)
                diagnostics.append(diagnostic)
        finite = np.logical_and.reduce([np.isfinite(predictions[m]).all(axis=1) for m in METHODS])
        common_score = finite & np.logical_and.reduce([flags[m]["score"] for m in METHODS])
        common_interior = common_score & np.logical_and.reduce([flags[m]["interior"] for m in METHODS])
        subsets = {"all_finite":np.ones(B, bool), "all_methods_score_eligible":common_score,
                   "all_methods_score_eligible_and_interior":common_interior}
        for subset, mask in subsets.items():
            for method in METHODS:
                selected = predictions[method].copy(); selected[~mask] = np.nan
                row = summarize_method(selected, full[method], B)
                row.update(variant=variant["id"],method=method,subset=subset,target=names[target],
                           target_scale="natural log ratio" if variant["dataset"] == "oliveoil" else "natural log length in millimetres",
                           score_passed=int(flags[method]["score"].sum()),
                           score_and_interior=int((flags[method]["score"]&flags[method]["interior"]).sum()),
                           full_prediction_available=bool(np.isfinite(full[method]).all()))
                summaries.append(row)
            for comparator in METHODS[1:]:
                L, R = predictions["shared_ld"].copy(), predictions[comparator].copy()
                L[~mask] = np.nan; R[~mask] = np.nan
                row = summarize_pair(L, R, full["shared_ld"], full[comparator], B)
                row.update(variant=variant["id"],left_method="shared_ld",right_method=comparator,
                           subset=subset,target=names[target])
                comparisons.append(row)
        arrays = {"analysis_grid":y,"target_index":np.array(target),"bootstrap_primary_task_indices":np.array([t["task_index"] for t in bootstrap_tasks])}
        for method in METHODS:
            arrays[method+"_bootstrap"] = predictions[method]
            arrays[method+"_full"] = full[method]
        np.savez_compressed(output/(variant["id"]+"_predictions.npz"), **arrays)
        grids.append({"variant":variant["id"],"grid_rows":n,"target_index":target,"coordinate_names":names,
                      "bootstrap_requested":B,"all_methods_finite":int(finite.sum()),
                      "all_methods_score_eligible":int(common_score.sum()),
                      "all_methods_score_eligible_and_interior":int(common_interior.sum())})
    pd.DataFrame(diagnostics).to_csv(output/"prediction_diagnostics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output/"stability_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output/"paired_stability.csv", index=False)
    write_json(output/"grid_and_eligibility.json", grids)
    if cv_summary:
        raw = inputs.bytes(cv_summary)
        (output/"existing_cv_scores.csv").write_bytes(raw)
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(), input_sha256=inputs.hashes,
        expected_predictions=len(PLAN["targets"])*len(METHODS)*(PLAN["bootstrap_repetitions"]+1),
        recorded_predictions=len(diagnostics),available_predictions=sum(r["prediction_available"] for r in diagnostics),
        outputs_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name != "analysis_plan.json"})
    write_json(output/"analysis_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--gaussian", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refined", type=Path)
    parser.add_argument("--cv-summary", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT/"docs/public_analysis_rules.json")
    args = parser.parse_args()
    result = run_analysis(args.primary,args.gaussian,args.output,args.protocol,args.refined,args.cv_summary)
    print(json.dumps({k:result[k] for k in ["procedure","expected_predictions","recorded_predictions","available_predictions"]}))


if __name__ == "__main__":
    main()
