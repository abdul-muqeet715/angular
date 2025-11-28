# atm_optimizer_full_fixed.py
# Single-file optimized ATM service optimizer (fixed simulate_candidate cost variable bug).
# - probability pre-screen
# - candidate generation (masks + cycles)
# - simulation (daily) with handling cost
# - profit scoring and top-K selection
#
# Author: ChatGPT (GPT-5 Thinking mini)
# Usage: configure `example` block at bottom and run.

import math
import itertools
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np

WEEKDAYS = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]

# ---------------------------
# Utilities
# ---------------------------
def mask_from_days(days: List[str]) -> int:
    m = 0
    for d in days:
        i = WEEKDAYS.index(d)
        m |= (1 << (6 - i))
    return m

def mask_to_days(mask: int) -> List[str]:
    s = f"{mask:07b}"
    return [WEEKDAYS[i] for i,ch in enumerate(s) if ch == '1']

def bits_of_mask(mask: int) -> List[int]:
    return [i for i in range(7) if ((mask >> (6-i)) & 1)]

def mask_from_bits(bits: List[int]) -> int:
    m = 0
    for i in bits:
        m |= (1 << (6 - i))
    return m

def all_nonempty_submasks(mask: int, max_bits: Optional[int] = None) -> List[int]:
    bits = bits_of_mask(mask)
    subs = []
    max_r = len(bits) if max_bits is None else min(len(bits), max_bits)
    for r in range(1, max_r+1):
        for comb in itertools.combinations(bits, r):
            subs.append(mask_from_bits(list(comb)))
    return sorted(set(subs))

def normal_cdf(x: float) -> float:
    # Standard normal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ---------------------------
# Demand builder
# ---------------------------
def build_demand_vector_from_history(start_date: datetime, end_date: datetime,
                                     history: Optional[List[float]] = None,
                                     forecast: Optional[List[float]] = None,
                                     default_mean: float = 8000.0,
                                     default_std_frac: float = 0.25,
                                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    days = (end_date - start_date).days + 1
    rng = np.random.default_rng(seed)
    if forecast is not None:
        if len(forecast) != days:
            raise ValueError("forecast length must equal horizon days")
        # Use forecast as mean, and default std_frac * mean as sigma if not provided
        forecast_mean = np.array(forecast, dtype=float)
        forecast_std = np.maximum(1.0, default_std_frac * forecast_mean)
        return forecast_mean, forecast_std
    if history is not None and len(history) >= 7:
        # bootstrap weekday-preserving
        fm = np.zeros(days, dtype=float)
        fs = np.zeros(days, dtype=float)
        for i in range(days):
            wd = (start_date + timedelta(days=i)).weekday()  # Mon=0..Sun=6 -> convert below
            sunwd = (wd + 1) % 7
            candidates = [history[j] for j in range(len(history)) if (j % 7) == sunwd]
            if not candidates:
                candidates = history
            choice = float(rng.choice(candidates))
            fm[i] = choice
            fs[i] = max(1.0, default_std_frac * choice)
        return fm, fs
    # fallback synthetic
    fm = rng.normal(default_mean, default_mean*0.15, days)
    fm = np.maximum(0.0, fm)
    fs = np.maximum(1.0, default_std_frac * fm)
    return fm, fs

# ---------------------------
# Cycle helpers
# ---------------------------
def generate_cycle_list(allowed_cycles: List[str]) -> List[Dict]:
    cycles = []
    for c in allowed_cycles:
        if isinstance(c, str) and c.upper().startswith('W'):
            n = int(c[1:])
            cycles.append({'type':'weekly','interval':n})
        elif isinstance(c, str) and c.upper().startswith('M'):
            w = int(c[1:])
            cycles.append({'type':'monthly_week','week':w})
        elif isinstance(c, str) and c.upper() == 'DAILY':
            cycles.append({'type':'daily','interval':1})
    return cycles

def is_scheduled(date: datetime, cycle: Dict, days_mask: int, start_date: datetime) -> bool:
    sunwd = (date.weekday() + 1) % 7
    if ((days_mask >> (6 - sunwd)) & 1) == 0:
        return False
    if cycle['type'] == 'daily':
        return True
    if cycle['type'] == 'weekly':
        week_index = ((date - start_date).days) // 7
        return (week_index % cycle['interval']) == 0
    if cycle['type'] == 'monthly_week':
        wk = ((date.day - 1) // 7) + 1
        return wk == cycle['week']
    return False

# ---------------------------
# Probability pre-screen (fast)
# ---------------------------
def prob_empty_in_range(start_balance: float, safety_stock: float,
                        forecast_mean: np.ndarray, forecast_std: np.ndarray,
                        start_idx: int, end_idx: int) -> float:
    """
    P( sum_{t=start_idx..end_idx} W_t >= start_balance - safety_stock )
    Approximates sum of independent normals as normal with mean=sum(mu), var=sum(sd^2).
    Returns probability in [0,1].
    """
    if end_idx < start_idx:
        return 0.0
    seg_mean = float(np.sum(forecast_mean[start_idx:end_idx+1]))
    seg_var = float(np.sum(np.square(forecast_std[start_idx:end_idx+1])))
    threshold = float(start_balance - safety_stock)
    if seg_var <= 0:
        return 1.0 if seg_mean >= threshold else 0.0
    sd = math.sqrt(seg_var)
    z = (threshold - seg_mean) / sd
    # P(sum >= threshold) = 1 - Phi(z)
    return float(max(0.0, min(1.0, 1.0 - normal_cdf(z))))

# ---------------------------
# Candidate generation (ADD / REPLACE / ADD+REPLACE)
# ---------------------------
def generate_candidates(service_mode: str,
                        add_allowed_days: List[str], repl_allowed_days: List[str], return_allowed_days: List[str],
                        add_cycles_allowed: List[str], repl_cycles_allowed: List[str],
                        max_mask_bits: int = 3) -> List[Dict]:
    """
    Returns list of candidate policy dicts. Each dict includes relevant masks & cycle dicts.
    For ADD: 'add_mask','add_cycle'
    For REPLACE: 'repl_mask','repl_cycle'
    For ADD_REPLACE: contains both
    Always includes 'return_mask' and 'return_cycle' placeholders (None if not used)
    """
    add_mask = mask_from_days(add_allowed_days) if add_allowed_days else 0
    repl_mask = mask_from_days(repl_allowed_days) if repl_allowed_days else 0
    return_mask = mask_from_days(return_allowed_days) if return_allowed_days else 0

    add_masks = all_nonempty_submasks(add_mask, max_bits=max_mask_bits) if add_mask else [0]
    repl_masks = all_nonempty_submasks(repl_mask, max_bits=max_mask_bits) if repl_mask else [0]
    return_masks = all_nonempty_submasks(return_mask, max_bits=max_mask_bits) if return_mask else [0]

    add_cycles = generate_cycle_list(add_cycles_allowed) if add_cycles_allowed else [{'type':'weekly','interval':1}]
    repl_cycles = generate_cycle_list(repl_cycles_allowed) if repl_cycles_allowed else [{'type':'weekly','interval':1}]
    # simple: use same cycles for return as add cycles in this implementation (can be extended)
    return_cycles = add_cycles

    candidates = []
    if service_mode == 'ADD':
        for am in add_masks:
            for ac in add_cycles:
                for rm in return_masks:
                    for rc in return_cycles:
                        candidates.append({'mode':'ADD','add_mask':am,'add_cycle':ac,'return_mask':rm,'return_cycle':rc})
    elif service_mode == 'REPLACE':
        for rm in repl_masks:
            for rc in repl_cycles:
                for retm in return_masks:
                    for rc2 in return_cycles:
                        candidates.append({'mode':'REPLACE','repl_mask':rm,'repl_cycle':rc,'return_mask':retm,'return_cycle':rc2})
    elif service_mode == 'ADD_REPLACE':
        # cross combine a subset of add and repl masks (limit explosion by small max_mask_bits)
        for am in add_masks:
            for ac in add_cycles:
                for rm in repl_masks:
                    for rc in repl_cycles:
                        for retm in return_masks:
                            for rc2 in return_cycles:
                                candidates.append({'mode':'ADD_REPLACE','add_mask':am,'add_cycle':ac,'repl_mask':rm,'repl_cycle':rc,'return_mask':retm,'return_cycle':rc2})
    else:
        raise ValueError("Unknown service_mode")
    return candidates

# ---------------------------
# Fast probability-based candidate filter
# ---------------------------
def prune_candidates_by_probability(candidates: List[Dict],
                                    start_date: datetime,
                                    forecast_mean: np.ndarray, forecast_std: np.ndarray,
                                    starting_balance: float, safety_stock: float,
                                    horizon_days: int,
                                    pre_withdrawal_pct: float,
                                    threshold: float = 0.25,
                                    check_cycles_ahead: int = 2) -> List[Dict]:
    """
    For each candidate, look at the first few upcoming scheduled service dates (check_cycles_ahead)
    and compute probability of emptying before that date. If any prob > threshold, drop candidate.
    This is a conservative prune.
    """
    survivors = []
    N = len(forecast_mean)
    for c in candidates:
        drop = False
        # We'll simulate upcoming scheduled dates by generating the schedule for next horizon_days
        # and find scheduled indices (days from 0) for the candidate
        scheduled_indices = []
        for day_idx in range(horizon_days):
            date = start_date + timedelta(days=day_idx)
            if c['mode'] == 'ADD':
                if is_scheduled(date, c['add_cycle'], c['add_mask'], start_date):
                    scheduled_indices.append(day_idx)
            elif c['mode'] == 'REPLACE':
                if is_scheduled(date, c['repl_cycle'], c['repl_mask'], start_date):
                    scheduled_indices.append(day_idx)
            else: # ADD_REPLACE
                if is_scheduled(date, c['add_cycle'], c['add_mask'], start_date) or is_scheduled(date, c['repl_cycle'], c['repl_mask'], start_date):
                    scheduled_indices.append(day_idx)
            if len(scheduled_indices) >= check_cycles_ahead:
                break
        # If no scheduled date within horizon, consider candidate risky and drop
        if len(scheduled_indices) == 0:
            # compute probability of emptying over whole horizon
            p = prob_empty_in_range(starting_balance, safety_stock, forecast_mean, forecast_std, 0, horizon_days-1)
            if p > threshold:
                drop = True
            else:
                drop = False
        else:
            # check each upcoming scheduled index
            last_checked = -1
            for idx in scheduled_indices:
                if idx <= last_checked:
                    continue
                # compute effective available cash before scheduled day:
                # we assume no service until that scheduled day -> check cumulative withdrawal
                p = prob_empty_in_range(starting_balance, safety_stock, forecast_mean, forecast_std, 0, idx)
                if p > threshold:
                    drop = True
                    break
                last_checked = idx
        if not drop:
            survivors.append(c)
    return survivors

# ---------------------------
# Detailed simulation (deterministic / Monte Carlo)
# ---------------------------
def simulate_candidate(candidate: Dict,
                       start_date: datetime, horizon_days: int,
                       forecast_mean: np.ndarray, forecast_std: np.ndarray,
                       max_holding: float, refill_amount: float, safety_stock: float,
                       pre_withdrawal_pct: float,
                       costs: Dict,
                       unplanned_days_mask: int,
                       interest_rate: float,
                       monte_carlo_runs: int = 1,
                       stockout_penalty: float = 10000.0,
                       rng_seed: int = 42) -> Dict:
    """
    Simulate candidate for horizon_days; returns aggregated metrics and profit.
    monte_carlo_runs>1 will jitter demand by per-day std and run avg results.
    """
    rng = np.random.default_rng(rng_seed)
    days = horizon_days
    # accumulate aggregates across runs
    agg_total_cost = []
    agg_handling_cost = []
    agg_unplanned_cost = []
    agg_stockouts = []
    agg_trips = []
    agg_end_cash = []

    # Safely define cost variables with defaults for all modes (avoid UnboundLocalError)
    planned_trip_cost = costs.get('planned_add_cost', 0.0)
    planned_return_cost = costs.get('planned_return_cost', 0.0)
    planned_combined_cost = costs.get('planned_combined_cost', planned_trip_cost + planned_return_cost)
    planned_add_cost = costs.get('planned_add_cost', 0.0)
    planned_replace_cost = costs.get('planned_replace_cost', 0.0)
    unplanned_trip_cost = costs.get('unplanned_add_cost', costs.get('unplanned_replace_cost', 0.0))
    unplanned_return_cost = costs.get('unplanned_return_cost', 0.0)

    # Override for specific modes (clarify intent)
    if candidate['mode'] == 'ADD':
        planned_trip_cost = costs.get('planned_add_cost', planned_trip_cost)
        planned_return_cost = costs.get('planned_return_cost', planned_return_cost)
        planned_combined_cost = costs.get('planned_combined_cost', planned_combined_cost)
        unplanned_trip_cost = costs.get('unplanned_add_cost', unplanned_trip_cost)
        unplanned_return_cost = costs.get('unplanned_return_cost', unplanned_return_cost)
    elif candidate['mode'] == 'REPLACE':
        planned_trip_cost = costs.get('planned_replace_cost', planned_trip_cost)
        planned_return_cost = costs.get('planned_return_cost', planned_return_cost)
        planned_combined_cost = costs.get('planned_combined_cost', planned_combined_cost)
        unplanned_trip_cost = costs.get('unplanned_replace_cost', unplanned_trip_cost)
        unplanned_return_cost = costs.get('unplanned_return_cost', unplanned_return_cost)
    else:  # ADD_REPLACE
        planned_add_cost = costs.get('planned_add_cost', planned_add_cost)
        planned_replace_cost = costs.get('planned_replace_cost', planned_replace_cost)
        planned_combined_cost = costs.get('planned_combined_cost', min(planned_add_cost + planned_replace_cost, planned_replace_cost))
        unplanned_trip_cost = costs.get('unplanned_add_cost', costs.get('unplanned_replace_cost', unplanned_trip_cost))
        unplanned_return_cost = costs.get('unplanned_return_cost', unplanned_return_cost)

    for run in range(monte_carlo_runs):
        cash = refill_amount  # start full
        total_service_cost = 0.0
        total_unplanned_cost = 0.0
        total_handling_cost = 0.0
        trips = 0
        stockouts = 0

        # build demand sample for this run
        if monte_carlo_runs > 1:
            # jitter forecast by normal noise using forecast_std
            noise = rng.normal(0.0, forecast_std)
            demand = np.maximum(0.0, forecast_mean + noise)
        else:
            demand = forecast_mean.copy()

        for day_idx in range(days):
            date = start_date + timedelta(days=day_idx)
            sunwd = (date.weekday() + 1) % 7
            # determine scheduled events for this candidate on this date
            scheduled_add = False
            scheduled_replace = False
            scheduled_return = False
            if candidate['mode'] == 'ADD':
                if is_scheduled(date, candidate['add_cycle'], candidate['add_mask'], start_date):
                    scheduled_add = True
                # return scheduling (if any)
                if candidate.get('return_mask') and is_scheduled(date, candidate['return_cycle'], candidate['return_mask'], start_date):
                    scheduled_return = True
            elif candidate['mode'] == 'REPLACE':
                if is_scheduled(date, candidate['repl_cycle'], candidate['repl_mask'], start_date):
                    scheduled_replace = True
                if candidate.get('return_mask') and is_scheduled(date, candidate['return_cycle'], candidate['return_mask'], start_date):
                    scheduled_return = True
            else: # ADD_REPLACE
                add_on = is_scheduled(date, candidate['add_cycle'], candidate['add_mask'], start_date)
                repl_on = is_scheduled(date, candidate['repl_cycle'], candidate['repl_mask'], start_date)
                # priority: REPLACE preferred if both scheduled (common), but decision can be dynamic — here choose REPLACE if scheduled
                if repl_on and add_on:
                    # dynamic rule: if upcoming demand (next 3 days) small enough, perform ADD instead of REPLACE
                    lookahead_days = 3
                    look_end = min(days-1, day_idx + lookahead_days)
                    upcoming_mean = float(np.sum(demand[day_idx:look_end+1]))
                    # threshold: if upcoming_mean < refill_amount*0.25 treat as small
                    if upcoming_mean < 0.25 * refill_amount:
                        scheduled_add = True
                    else:
                        scheduled_replace = True
                else:
                    scheduled_add = add_on
                    scheduled_replace = repl_on
                if candidate.get('return_mask') and is_scheduled(date, candidate['return_cycle'], candidate['return_mask'], start_date):
                    scheduled_return = True

            # Determine and apply service cost(s) robustly (ensure variables exist)
            # We'll use explicit variables defined above: planned_add_cost, planned_replace_cost, planned_trip_cost, planned_return_cost, planned_combined_cost
            # default service_type used only for accounting
            if scheduled_replace and scheduled_add:
                # both scheduled
                # If ADD_REPLACE candidate, use planned_combined_cost if provided else choose replace
                if candidate['mode'] == 'ADD_REPLACE':
                    service_cost_here = planned_combined_cost
                    # perform replace semantics (swap to full)
                    cash = max_holding
                else:
                    # for pure ADD or REPLACE modes this path shouldn't usually happen, but handle gracefully
                    service_cost_here = planned_trip_cost
                    if candidate['mode'] == 'ADD':
                        cash = min(cash + refill_amount, max_holding)
                    else:
                        cash = max_holding
                total_service_cost += service_cost_here
                trips += 1
            elif scheduled_replace:
                # Replace action
                # For ADD_REPLACE mode use planned_replace_cost, else planned_trip_cost was set appropriately earlier
                service_cost_here = planned_replace_cost if candidate['mode']=='ADD_REPLACE' else planned_trip_cost
                total_service_cost += service_cost_here
                trips += 1
                cash = max_holding
            elif scheduled_add:
                # Add action
                service_cost_here = planned_add_cost if candidate['mode']=='ADD_REPLACE' else planned_trip_cost
                total_service_cost += service_cost_here
                trips += 1
                cash = min(cash + refill_amount, max_holding)
            elif scheduled_return:
                service_cost_here = planned_return_cost
                total_service_cost += service_cost_here
                trips += 1
                # assume return slightly reduces cash (domain-specific). left as-is for now.
            # else: no scheduled service this morning

            # pre-withdrawal (percentage of that day's withdrawal)
            withdrawal = float(demand[day_idx])
            pre = pre_withdrawal_pct * withdrawal
            take_pre = min(pre, withdrawal)
            remaining = withdrawal - take_pre
            # apply withdrawals
            cash -= take_pre
            cash -= remaining

            # compute daily handling cost on closing balance (after withdrawals and after any services performed that morning)
            # Note: in many operations, service occurs early morning -> here we modeled service before withdrawal for scheduled items.
            handling_cost_today = max(0.0, cash) * (interest_rate / 365.0)
            total_handling_cost += handling_cost_today

            # safety check
            if cash < safety_stock:
                # emergency allowed?
                if ((unplanned_days_mask >> (6 - sunwd)) & 1):
                    total_unplanned_cost += unplanned_trip_cost
                    trips += 1
                    cash = min(cash + refill_amount, max_holding)
                else:
                    # find next allowed unplanned day in next 7 days, else stockout
                    found = False
                    for look in range(1,8):
                        nd = date + timedelta(days=look)
                        nd_sunwd = (nd.weekday() + 1) % 7
                        if ((unplanned_days_mask >> (6 - nd_sunwd)) & 1):
                            total_unplanned_cost += unplanned_trip_cost
                            trips += 1
                            cash = min(cash + refill_amount, max_holding)
                            found = True
                            break
                    if not found:
                        stockouts += 1

            if cash < 0:
                stockouts += 1

        # finalize run aggregates
        total_cost_run = total_service_cost + total_unplanned_cost + total_handling_cost + stockouts * stockout_penalty
        agg_total_cost.append(total_cost_run)
        agg_handling_cost.append(total_handling_cost)
        agg_unplanned_cost.append(total_unplanned_cost)
        agg_stockouts.append(stockouts)
        agg_trips.append(trips)
        agg_end_cash.append(cash)

    # aggregate stats
    res = {
        'candidate': candidate,
        'mode': candidate['mode'],
        'mean_total_cost': float(np.mean(agg_total_cost)),
        'std_total_cost': float(np.std(agg_total_cost)),
        'mean_handling_cost': float(np.mean(agg_handling_cost)),
        'mean_unplanned_cost': float(np.mean(agg_unplanned_cost)),
        'mean_stockouts': float(np.mean(agg_stockouts)),
        'mean_trips': float(np.mean(agg_trips)),
        'mean_end_cash': float(np.mean(agg_end_cash)),
        # profit definition: negative total cost (so larger is better)
        'profit_score': float(-np.mean(agg_total_cost))
    }
    return res

# ---------------------------
# Runner: full pipeline
# ---------------------------
def optimize_atm_service(start_date: datetime, end_date: datetime,
                         history: Optional[List[float]], forecast: Optional[List[float]],
                         service_mode: str,
                         add_allowed_days: List[str], repl_allowed_days: List[str], return_allowed_days: List[str],
                         add_cycles_allowed: List[str], repl_cycles_allowed: List[str],
                         user_params: Dict,
                         algo_params: Dict = None) -> List[Dict]:
    """
    Returns top-K candidate results sorted by profit_score (descending).
    user_params must include:
      - max_holding, safety_stock, refill_amount, pre_withdrawal_pct,
      - costs dict (see example), unplanned_days list, interest_rate,
      - starting_balance (optional, defaults to refill_amount)
    algo_params can include:
      - horizon_days (int), max_mask_bits, prob_threshold, top_k, monte_carlo_runs
    """
    if algo_params is None:
        algo_params = {}
    horizon_days = algo_params.get('horizon_days', (end_date - start_date).days + 1)
    max_mask_bits = algo_params.get('max_mask_bits', 3)
    prob_threshold = algo_params.get('prob_threshold', 0.25)
    check_cycles_ahead = algo_params.get('check_cycles_ahead', 2)
    top_k = algo_params.get('top_k', 5)
    monte_carlo_runs = algo_params.get('monte_carlo_runs', 5)
    stockout_penalty = algo_params.get('stockout_penalty', 10000.0)
    seed = algo_params.get('seed', 42)

    # build demand
    forecast_mean, forecast_std = build_demand_vector_from_history(start_date, end_date, history, forecast, seed=seed)
    # starting balance
    starting_balance = user_params.get('starting_balance', user_params.get('refill_amount', 0.0))
    # construct unplanned mask
    unplanned_mask = mask_from_days(user_params.get('unplanned_days', ['Mon','Tue','Wed','Thu','Fri']))

    # generate candidate set
    candidates = generate_candidates(service_mode,
                                     add_allowed_days, repl_allowed_days, return_allowed_days,
                                     add_cycles_allowed, repl_cycles_allowed,
                                     max_mask_bits=max_mask_bits)

    # quick prune using probability
    survivors = prune_candidates_by_probability(candidates,
                                                start_date,
                                                forecast_mean, forecast_std,
                                                starting_balance, user_params['safety_stock'],
                                                horizon_days,
                                                user_params['pre_withdrawal_pct'],
                                                threshold=prob_threshold,
                                                check_cycles_ahead=check_cycles_ahead)

    # if no survivors, widen threshold (graceful fallback)
    if len(survivors) == 0:
        survivors = candidates  # fallback to all

    # simulate survivors
    results = []
    costs = user_params['costs']
    for c in survivors:
        res = simulate_candidate(c,
                                 start_date, horizon_days,
                                 forecast_mean, forecast_std,
                                 user_params['max_holding'], user_params['refill_amount'], user_params['safety_stock'],
                                 user_params['pre_withdrawal_pct'],
                                 costs,
                                 unplanned_mask,
                                 user_params['interest_rate'],
                                 monte_carlo_runs=monte_carlo_runs,
                                 stockout_penalty=stockout_penalty,
                                 rng_seed=seed)
        results.append(res)

    # sort by profit_score descending (profit defined as -total_cost)
    results_sorted = sorted(results, key=lambda r: r['profit_score'], reverse=True)
    return results_sorted[:top_k]

# ---------------------------
# Helper: save top-1 horizon CSV
# ---------------------------
def save_horizon_csv(top_result: Dict, start_date: datetime, horizon_days: int,
                     forecast_mean: np.ndarray, filename: str, user_params: Dict):
    """
    Saves a CSV of daily timeline for top_result candidate (deterministic using forecast_mean).
    Columns: DayIndex,Date,Opening,Withdrawal,PreWithdrawal,Closing,HandlingCost,ServiceType,ServiceCost,UnplannedFlag
    """
    candidate = top_result['candidate']
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['DayIndex','Date','Opening','Withdrawal','PreWithdrawal','Closing','HandlingCost','ServiceType','ServiceCost','UnplannedFlag'])
        cash = user_params['refill_amount']
        for i in range(horizon_days):
            date = start_date + timedelta(days=i)
            sunwd = (date.weekday() + 1) % 7
            opening = cash
            # scheduled?
            service_type = 'NONE'
            service_cost = 0.0
            if candidate['mode'] == 'ADD':
                if is_scheduled(date, candidate['add_cycle'], candidate['add_mask'], start_date):
                    service_type = 'ADD'
                    service_cost = user_params['costs']['planned_add_cost']
                    cash = min(cash + user_params['refill_amount'], user_params['max_holding'])
            elif candidate['mode'] == 'REPLACE':
                if is_scheduled(date, candidate['repl_cycle'], candidate['repl_mask'], start_date):
                    service_type = 'REPLACE'
                    service_cost = user_params['costs']['planned_replace_cost']
                    cash = user_params['max_holding']
            else:
                add_on = is_scheduled(date, candidate['add_cycle'], candidate['add_mask'], start_date)
                repl_on = is_scheduled(date, candidate['repl_cycle'], candidate['repl_mask'], start_date)
                if repl_on and add_on:
                    # choose replace by default
                    service_type = 'REPLACE'
                    service_cost = user_params['costs'].get('planned_combined_cost', user_params['costs'].get('planned_replace_cost',0.0))
                    cash = user_params['max_holding']
                elif repl_on:
                    service_type = 'REPLACE'
                    service_cost = user_params['costs']['planned_replace_cost']
                    cash = user_params['max_holding']
                elif add_on:
                    service_type = 'ADD'
                    service_cost = user_params['costs']['planned_add_cost']
                    cash = min(cash + user_params['refill_amount'], user_params['max_holding'])
            withdrawal = float(forecast_mean[i])
            pre = user_params['pre_withdrawal_pct'] * withdrawal
            pre_taken = min(pre, withdrawal)
            remaining = withdrawal - pre_taken
            cash -= pre_taken
            cash -= remaining
            handling = max(0.0, cash) * (user_params['interest_rate'] / 365.0)
            unplanned_flag = ''
            if cash < user_params['safety_stock']:
                # emergency if allowed this weekday
                if ((mask_from_days(user_params['unplanned_days']) >> (6 - sunwd)) & 1):
                    unplanned_flag = 'UNPLANNED'
                    service_cost += user_params['costs'].get('unplanned_add_cost', 0.0)
                    cash = min(cash + user_params['refill_amount'], user_params['max_holding'])
                else:
                    unplanned_flag = 'POTENTIAL_STOCKOUT'
            w.writerow([i, date.strftime('%Y-%m-%d'), round(opening,2), round(withdrawal,2), round(pre_taken,2),
                        round(cash,2), round(handling,2), service_type, round(service_cost,2), unplanned_flag])
    print(f"Horizon saved to {filename}")

# ---------------------------
# Example usage
# ---------------------------
if __name__ == '__main__':
    # configure dates
    start_date = datetime.today()
    end_date = start_date + timedelta(days=44)  # 45-day horizon

    # sample history (optional) - if you have forecast, pass it in forecast variable
    rng = np.random.default_rng(123)
    history = np.maximum(0, np.round(rng.normal(9000, 2000, 90))).tolist()

    # user params
    user_params = {
        'max_holding': 250000.0,
        'safety_stock': 30000.0,
        'pre_withdrawal_pct': 0.05,  # percentage of that day's withdrawal
        'refill_amount': 200000.0,
        'starting_balance': 200000.0,
        'unplanned_days': ['Mon','Tue','Wed','Thu','Fri'],
        'interest_rate': 0.06,  # 6% annual cost (holding/opportunity)
        'costs': {
            # ADD mode costs
            'planned_add_cost': 350.0,
            'planned_return_cost': 150.0,
            'planned_combined_cost': 450.0,
            'unplanned_add_cost': 800.0,
            'unplanned_return_cost': 600.0,
            # REPLACE costs (if used)
            'planned_replace_cost': 380.0,
            'unplanned_replace_cost': 900.0
        }
    }

    # scheduling constraints
    service_mode = 'ADD_REPLACE'  # 'ADD' | 'REPLACE' | 'ADD_REPLACE'
    add_allowed_days = ['Mon','Tue','Fri']
    repl_allowed_days = ['Mon','Wed','Sat']
    return_allowed_days = ['Sat','Sun']  # example
    add_cycles_allowed = ['W1','W2','W3','W4']  # weekly, bi-weekly...
    repl_cycles_allowed = ['W1','W2','W3','W4']

    algo_params = {
        'horizon_days': 45,
        'max_mask_bits': 3,
        'prob_threshold': 0.25,
        'check_cycles_ahead': 2,
        'top_k': 5,
        'monte_carlo_runs': 10,
        'stockout_penalty': 15000.0,
        'seed': 42
    }

    # run optimizer
    results = optimize_atm_service(start_date, end_date, history, None,
                                   service_mode,
                                   add_allowed_days, repl_allowed_days, return_allowed_days,
                                   add_cycles_allowed, repl_cycles_allowed,
                                   user_params, algo_params)

    # show top results (print fields exist in returned dict)
    for i,r in enumerate(results):
        print(f"Rank {i+1}: mode={r['mode']}, profit_score={r['profit_score']:.2f}, mean_cost={-r['profit_score']:.2f}")
        cand = r['candidate']
        if cand['mode']=='ADD':
            print("  ADD days:", mask_to_days(cand['add_mask']), "cycle:", cand['add_cycle'])
        elif cand['mode']=='REPLACE':
            print("  REPL days:", mask_to_days(cand['repl_mask']), "cycle:", cand['repl_cycle'])
        else:
            print("  ADD days:", mask_to_days(cand['add_mask']), "cycle:", cand['add_cycle'],
                  "| REPL days:", mask_to_days(cand['repl_mask']), "cycle:", cand['repl_cycle'])
        print("  mean_trips:", r['mean_trips'], "mean_stockouts:", r['mean_stockouts'], "mean_handling_cost:", r['mean_handling_cost'])

    # save horizon for top-1
    if results:
        # rebuild demand deterministic for CSV
        fm, fs = build_demand_vector_from_history(start_date, end_date, history, None, seed=42)
        save_horizon_csv(results[0], start_date, algo_params['horizon_days'], fm, 'top1_horizon.csv', user_params)
