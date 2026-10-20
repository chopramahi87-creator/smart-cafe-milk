# app.py - Milk Shop: Daily Manager Dashboard (Q-learning + demand forecast)
# Run:  streamlit run app.py      (cafe.py and agent.py must be in the same folder)
import json, math, os, random, time, datetime as dt
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Milk Shop - Daily Manager", page_icon="🥛", layout="wide")
try:
    from cafe import CafeSim
    from agent import train, policy_from_Q
except Exception as e:
    st.error("Could not import cafe.py / agent.py"); st.exception(e); st.stop()

BASE = os.path.dirname(os.path.abspath(__file__))
LOG, MODEL = os.path.join(BASE, "daily_log.csv"), os.path.join(BASE, "model.npz")
COLS = ["date", "opening", "recommended", "ordered", "customers", "sold", "unsold",
        "spoiled", "stockout", "carry", "profit", "source"]
NAVY, SLATE, GREEN, RED = "#1e3a8a", "#94a3b8", "#15803d", "#b91c1c"
today = dt.date.today()

st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1200px}
[data-testid="stMetric"]{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px}
[data-testid="stMetric"] *{color:#0f172a !important}
.rec{background:#1e3a8a;color:#fff;border-radius:12px;padding:20px 26px;margin-bottom:8px}
.rec .big{font-size:3rem;font-weight:700;line-height:1.1}.rec .sm{opacity:.85}
</style>""", unsafe_allow_html=True)

# ---------------- settings (from the assignment's business rules) ----------------
sb = st.sidebar
sb.title("Business settings")
P = dict(
    price=sb.number_input("Selling price / unit (AED)", 0.0, 100.0, 5.0, 0.5),
    order_cost=sb.number_input("Purchase cost / unit (AED)", 0.0, 100.0, 1.0, 0.1),
    holding_cost=sb.number_input("Holding cost / leftover unit (AED)", 0.0, 10.0, 0.3, 0.1),
    spoilage_cost=sb.number_input("Spoilage cost / spoiled unit (AED)", 0.0, 20.0, 1.5, 0.1),
    stockout_penalty=sb.number_input("Stockout penalty / customer (AED)", 0.0, 50.0, 6.0, 0.5),
    spoil_frac=sb.slider("Share of leftover that spoils overnight", 0.0, 1.0, 0.25, 0.05),
    mean_demand=sb.number_input("Typical daily customers (starting estimate)", 1, 100, 7, 1),
    max_stock=sb.number_input("Fridge capacity (units)", 5, 100, 15, 1),
)
P["max_stock"] = int(P["max_stock"]); P["mean_demand"] = int(P["mean_demand"]); BUSY = 0.20
sb.caption("1 customer = 1 unit of milk. Defaults come from the project brief. Change them to your real costs, then retrain the AI.")

def make_env():
    return CafeSim(**P)

# ---------------- business maths ----------------
def day_calc(P, stock, order, demand, sold=None, spoiled=None):
    order = int(np.clip(order, 0, P["max_stock"]))
    avail = min(stock + order, P["max_stock"])
    sold = min(avail, demand) if sold is None else min(int(sold), avail)
    stockout = max(demand - sold, 0); unsold = avail - sold
    spoiled = int(round(P["spoil_frac"] * unsold)) if spoiled is None else min(int(spoiled), unsold)
    carry = unsold - spoiled
    profit = sold * P["price"] - (order * P["order_cost"] + unsold * P["holding_cost"]
             + spoiled * P["spoilage_cost"] + stockout * P["stockout_penalty"])
    return round(profit, 2), carry, dict(opening=stock, ordered=order, customers=demand, sold=sold,
            unsold=unsold, spoiled=spoiled, stockout=stockout, carry=carry, profit=round(profit, 2))

def pois(m, n):
    p = np.array([math.exp(-m + k * math.log(max(m, 1e-9)) - math.lgamma(k + 1)) for k in range(n)])
    return p / p.sum()

def smart_order(P, Q, stock, mean, gamma):
    """Expected value of each order = today's expected profit + discounted future value (Q-table),
    using tomorrow's forecast demand (same demand shape as the simulator)."""
    n = int(mean * 3 + 20); m0 = max(0.5, mean - 5 * BUSY)
    pm = (1 - BUSY) * pois(m0, n) + BUSY * pois(m0 + 5, n)
    scores = []
    for o in range(0, P["max_stock"] - stock + 1):
        e = 0.0
        for d in range(n):
            pr, carry, _ = day_calc(P, stock, o, d)
            e += pm[d] * (pr + gamma * Q[carry].max())
        scores.append(e)
    return int(np.argmax(scores)), scores

# ---------------- data log ----------------
def load_log():
    df = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df: df[c] = np.nan
    return df[COLS].sort_values("date").reset_index(drop=True)

def upsert(df, date, **vals):
    d = str(date)
    if (df["date"] == d).any():
        for k, v in vals.items(): df.loc[df["date"] == d, k] = v
    else:
        row = {c: np.nan for c in COLS}; row.update(date=d, **vals)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True); df.to_csv(LOG, index=False); return df

def done_rows(df, before=None):
    d = df.dropna(subset=["customers"])
    return d[d["date"] < str(before)] if before else d

def forecast(hist, target, default_mean):
    h = hist.dropna(subset=["customers"]).copy()
    if len(h) < 3:
        m = float(default_mean); s = math.sqrt(m) * 1.3
    else:
        c = h["customers"].astype(float).reset_index(drop=True)
        ew = c.ewm(span=7).mean(); m = ew.iloc[-1]
        if len(h) >= 14:
            h["dow"] = pd.to_datetime(h["date"]).dt.weekday
            g = h.groupby("dow")["customers"].mean(); wd = pd.Timestamp(target).weekday()
            if wd in g.index:
                m *= min(max(1 + 0.5 * (g[wd] / c.mean() - 1), 0.75), 1.25)
        res = (c - ew.shift(1)).dropna().tail(14)
        s = max(math.sqrt(m), float(res.std()) if len(res) > 2 else 0.0)
    return float(m), max(0.0, m - 1.28 * s), float(m + 1.28 * s)

def backtest_mae(df, default_mean, k=14):
    d = done_rows(df); errs = []
    for _, r in d.tail(k).iterrows():
        m, _, _ = forecast(d[d["date"] < r["date"]], r["date"], default_mean)
        errs.append(abs(m - r["customers"]))
    return float(np.mean(errs)) if len(errs) >= 3 else None

# ---------------- AI model ----------------
def load_model():
    if "model" not in st.session_state and os.path.exists(MODEL):
        z = np.load(MODEL, allow_pickle=True)
        st.session_state["model"] = dict(Q=z["Q"], curve=list(z["curve"]),
                                         P=json.loads(str(z["P"])), gamma=float(z["gamma"]), id=time.time())
    return st.session_state.get("model")

def eval_policy(env, fn, months=300):
    prof = []; dem = sold = spo = so = orders = 0
    for _ in range(months):
        s = env.reset(); done = False; t = 0
        while not done:
            r, s, done, i = env.step(fn(s)); t += r
            dem += i["demand"]; sold += i["sold"]; spo += i["spoiled"]; so += i["stockout"]; orders += i["order"]
        prof.append(t)
    return np.mean(prof), sold / max(dem, 1), spo / max(orders, 1), so / max(dem, 1)

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=.25, axis="y")

# ---------------- page ----------------
st.title("Milk Shop - Daily Manager")
if "flash" in st.session_state: st.success(st.session_state.pop("flash"))
log = load_log(); model = load_model()
stale = bool(model) and any(abs(model["P"][k] - P[k]) > 1e-9 for k in P)
if model is None:
    st.warning("The AI has not been trained yet. Open the **AI Training** tab and press Train (takes a few seconds).")
elif stale:
    st.warning("Business settings changed since the AI was trained. Retrain in the **AI Training** tab for correct advice.")
policy = policy_from_Q(model["Q"]) if model else None
gamma = model["gamma"] if model else 0.95

tabs = st.tabs(["Morning Order", "End of Day", "Tomorrow's Forecast", "History & Reports",
                "AI Training", "Practice: You vs AI"])

# ===== 1. Morning order =====
with tabs[0]:
    st.subheader("Morning check - how much should we order today?")
    a, b = st.columns([1, 2])
    day = a.date_input("Date", today, key="mdate")
    prev = done_rows(log, day)
    open_default = int(prev.iloc[-1]["carry"]) if len(prev) else 5
    opening = a.number_input("Milk in the fridge this morning (units)", 0, P["max_stock"],
                             min(open_default, P["max_stock"]), key=f"open_{day}",
                             help="Pre-filled from yesterday's carry-over. Correct it with the real count.")
    existing = log[log["date"] == str(day)]
    mean, lo, hi = forecast(prev, day, P["mean_demand"])
    if model is None:
        b.info("Train the AI first.")
    else:
        cap = P["max_stock"] - opening
        rec, scores = smart_order(P, model["Q"], opening, mean, gamma) if cap > 0 else (0, [0])
        base = min(int(policy[opening]), cap)
        b.markdown(f'<div class="rec"><div class="sm">RECOMMENDED ORDER FOR {pd.Timestamp(day):%A, %d %b %Y}</div>'
                   f'<div class="big">{rec} units</div>'
                   f'<div class="sm">Expected customers today: {mean:.0f} (likely {lo:.0f} to {hi:.0f})</div></div>',
                   unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Milk in fridge now", f"{opening} units")
        m2.metric("Stock after order arrives", f"{opening + rec} units")
        m3.metric("Space left in fridge", f"{cap - rec} units")
        st.caption(f"Basic AI strategy alone (without today's forecast) would suggest {base} units. "
                   f"The recommendation above also uses the demand forecast.")
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.bar(range(len(scores)), scores, color=[NAVY if i == rec else SLATE for i in range(len(scores))])
        ax.set_xlabel("Units ordered"); ax.set_ylabel("Expected value (AED)")
        ax.set_title("Expected value of each possible order (highlighted = recommended)"); style(ax)
        st.pyplot(fig); plt.close(fig)
        st.markdown("**Manager checklist**  \n1. Count the milk actually in the fridge and correct the number above.  \n"
                    "2. Place the order (recommended or your own judgement).  \n3. Record the quantity below.")
        with st.form("morning"):
            ordered = st.number_input("Quantity actually ordered today", 0, P["max_stock"],
                                      int(existing.iloc[0]["ordered"]) if len(existing) and pd.notna(existing.iloc[0]["ordered"]) else rec)
            if st.form_submit_button("Save today's order", type="primary"):
                upsert(log, day, opening=opening, recommended=rec, ordered=ordered, source="live")
                st.session_state["flash"] = f"Saved: {ordered} units ordered for {day}. Enter the results in the End of Day tab tonight."
                st.rerun()

# ===== 2. End of day =====
with tabs[1]:
    st.subheader("End of day - record what happened")
    pend = log.dropna(subset=["ordered"])
    if pend.empty:
        st.info("Save a morning order first.")
    else:
        dates = list(pend["date"][::-1])
        sel = st.selectbox("Day", dates, index=dates.index(str(today)) if str(today) in dates else 0)
        r = pend[pend["date"] == sel].iloc[0]
        fm = forecast(done_rows(log, sel), sel, P["mean_demand"])[0]
        st.write(f"Opening stock **{int(r['opening'])}**, ordered **{int(r['ordered'])}**.")
        cust = st.number_input("Customers who visited today (wanted milk)", 0, 500,
                               int(r["customers"]) if pd.notna(r["customers"]) else int(round(fm)), key=f"cust_{sel}")
        ov = st.checkbox("I counted sold / spoiled units myself (override the automatic figures)", key=f"ov_{sel}")
        sold_in = spo_in = None
        if ov:
            c1, c2 = st.columns(2)
            sold_in = c1.number_input("Units sold", 0, P["max_stock"], min(int(r["opening"] + r["ordered"]), cust), key=f"s_{sel}")
            spo_in = c2.number_input("Units spoiled / thrown away", 0, P["max_stock"], 0, key=f"sp_{sel}")
        pr, carry, i = day_calc(P, int(r["opening"]), int(r["ordered"]), int(cust), sold_in, spo_in)
        k = st.columns(6)
        k[0].metric("Customers visited", i["customers"]); k[1].metric("Milk sold", i["sold"])
        k[2].metric("Milk NOT sold", i["unsold"]); k[3].metric("Spoiled", i["spoiled"])
        k[4].metric("Customers not served", i["stockout"]); k[5].metric("Profit", f"AED {pr:,.1f}")
        if i["stockout"]: st.error(f"{i['stockout']} customer(s) left without milk. Consider ordering more on similar days.")
        elif i["unsold"] >= 4: st.warning(f"{i['unsold']} units were left unsold. Part of it will spoil overnight.")
        else: st.success("Well balanced day: few leftovers and everyone served.")
        if st.button("Save end-of-day results", type="primary"):
            upsert(log, sel, customers=i["customers"], sold=i["sold"], unsold=i["unsold"], spoiled=i["spoiled"],
                   stockout=i["stockout"], carry=carry, profit=pr)
            st.session_state["flash"] = f"Saved results for {sel}. Carry-over for tomorrow: {carry} units."
            st.rerun()

# ===== 3. Forecast =====
with tabs[2]:
    st.subheader("Customer forecast")
    d = done_rows(log)
    tom = today + dt.timedelta(days=1)
    m, lo, hi = forecast(d, tom, P["mean_demand"])
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Expected customers - {pd.Timestamp(tom):%A %d %b}", f"{m:.0f}")
    c2.metric("Likely range (80%)", f"{lo:.0f} to {hi:.0f}")
    mae = backtest_mae(log, P["mean_demand"])
    c3.metric("Typical forecast error", f"+/- {mae:.1f} customers" if mae is not None else "Need 6+ days of data")
    if len(d) < 14:
        st.info(f"Forecast is based on {len(d)} recorded day(s). It improves after 14+ days, when weekday patterns are included.")
    if len(d):
        t = d.tail(30)
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(pd.to_datetime(t["date"]), t["customers"], color=NAVY, marker="o", ms=3, label="Actual customers")
        ax.errorbar([pd.Timestamp(tom)], [m], yerr=[[m - lo], [hi - m]], fmt="s", color=GREEN, capsize=5, label="Forecast (80% range)")
        ax.set_ylabel("Customers"); ax.legend(frameon=False); style(ax); fig.autofmt_xdate()
        st.pyplot(fig); plt.close(fig)
    rows = []
    for n in range(1, 8):
        dd = today + dt.timedelta(days=n); mm, l, h = forecast(d, dd, P["mean_demand"])
        rows.append({"Date": f"{dd:%a %d %b}", "Expected customers": round(mm), "Low": round(l), "High": round(h)})
    st.markdown("**Next 7 days**"); st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Method: recent trend (exponentially weighted average of the last days) adjusted for day-of-week once enough history exists.")

# ===== 4. History =====
with tabs[3]:
    st.subheader("History & reports")
    d = done_rows(log)
    if d.empty:
        st.info("No completed days yet. Record days in the Morning/End of Day tabs, or load sample data to explore.")
    else:
        w = d.tail(30)
        tot_c = w["customers"].sum()
        k = st.columns(5)
        k[0].metric("Profit (last 30 days)", f"AED {w['profit'].sum():,.0f}")
        k[1].metric("Avg customers / day", f"{w['customers'].mean():.1f}")
        k[2].metric("Service level", f"{w['sold'].sum() / max(tot_c, 1) * 100:.0f}%")
        k[3].metric("Avg unsold / day", f"{w['unsold'].mean():.1f}")
        k[4].metric("Spoiled units", f"{int(w['spoiled'].sum())}")
        fig, ax = plt.subplots(figsize=(10, 3))
        x = pd.to_datetime(w["date"])
        ax.bar(x, w["sold"], color=NAVY, label="Sold"); ax.bar(x, w["unsold"], bottom=w["sold"], color=SLATE, label="Unsold")
        ax.plot(x, w["customers"], color=RED, marker="o", ms=3, label="Customers"); ax.legend(frameon=False, ncol=3)
        ax.set_ylabel("Units"); style(ax); fig.autofmt_xdate(); st.pyplot(fig); plt.close(fig)
        if (log["source"] == "demo").any(): st.warning("This history contains SAMPLE data. Clear it before using real records.")
        st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Download log (CSV)", log.to_csv(index=False), "milk_shop_log.csv", "text/csv")
    c1, c2 = st.columns(2)
    if c1.button("Load 35 days of sample data", disabled=bool((log["source"] == "live").any())):
        e = make_env(); stock = 5; df = log
        for n in range(35, 0, -1):
            o = min(random.randint(5, 10), P["max_stock"] - stock)
            pr, carry, i = day_calc(P, stock, o, e._demand())
            df = upsert(df, today - dt.timedelta(days=n), opening=stock, ordered=o, customers=i["customers"], sold=i["sold"],
                        unsold=i["unsold"], spoiled=i["spoiled"], stockout=i["stockout"], carry=carry, profit=pr, source="demo")
            stock = carry
        st.session_state["flash"] = "Sample data loaded."; st.rerun()
    if c2.checkbox("I understand: delete ALL records") and c2.button("Delete all records"):
        if os.path.exists(LOG): os.remove(LOG)
        st.session_state["flash"] = "All records deleted."; st.rerun()

# ===== 5. AI training =====
with tabs[4]:
    st.subheader("Train and inspect the AI manager")
    c1, c2 = st.columns(2)
    episodes = c1.select_slider("Training months (episodes)", [2000, 5000, 8000, 12000, 15000, 20000], value=12000)
    g = c2.slider("Gamma (how much it values the future)", 0.0, 0.99, 0.95, 0.01)
    if st.button("Train the AI", type="primary"):
        with st.spinner("Training on simulated months..."):
            Q, curve = train(make_env(), episodes=episodes, gamma=g)
            np.savez(MODEL, Q=Q, curve=np.array(curve), P=json.dumps(P), gamma=g)
            st.session_state.pop("model", None); st.session_state.pop("score", None)
        st.session_state["flash"] = "Training complete and saved. No need to retrain each day."; st.rerun()
    if model:
        key = (json.dumps(P, sort_keys=True), model["id"])
        if st.session_state.get("score", (None,))[0] != key:
            env = make_env(); ao = lambda s: int(policy[int(np.clip(s, 0, P["max_stock"]))])
            st.session_state["score"] = (key, eval_policy(env, ao), eval_policy(env, lambda s: 4)[0],
                                         eval_policy(env, lambda s: 10)[0],
                                         eval_policy(env, lambda s: random.randint(0, P["max_stock"]))[0])
        _, (rl, svc, spr, sor), cau, gen, rnd = st.session_state["score"]
        k = st.columns(4)
        k[0].metric("AI avg monthly profit", f"AED {rl:,.0f}"); k[1].metric("Service level", f"{svc*100:.0f}%")
        k[2].metric("Spoilage rate", f"{spr*100:.0f}%"); k[3].metric("Stockout rate", f"{sor*100:.0f}%")
        curve = model["curve"]; w = max(1, len(curve) // 100)
        sm = np.convolve(curve, np.ones(w) / w, mode="valid")
        l, r = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(sm, color=NAVY); ax.set_title("Learning curve")
        ax.set_xlabel("Training month"); ax.set_ylabel("Profit (AED)"); style(ax); l.pyplot(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3)); ax.bar(range(len(policy)), policy, color=NAVY)
        ax.set_title("Learned strategy"); ax.set_xlabel("Stock on hand"); ax.set_ylabel("Units to order"); style(ax)
        r.pyplot(fig); plt.close(fig)
        names = ["Cautious (always 4)", "Random", "Generous (always 10)", "RL Agent"]; vals = [cau, rnd, gen, rl]
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.barh(names, vals, color=[SLATE, SLATE, SLATE, GREEN])
        for i, v in enumerate(vals): ax.text(v, i, f" {v:,.0f}", va="center")
        ax.set_xlabel("Average monthly profit (AED)"); ax.set_title("Scoreboard: AI vs guessing"); style(ax)
        st.pyplot(fig); plt.close(fig)
        st.caption(f"Started around AED {np.mean(curve[:200]):.0f}/month, ended around AED {np.mean(curve[-200:]):.0f}/month.")

# ===== 6. Practice game =====
with tabs[5]:
    st.subheader("Practice: you manage 10 days, then the AI plays the same days")
    if not model: st.info("Train the AI first.")
    else:
        if "gd" not in st.session_state or st.button("New 10-day challenge"):
            e = make_env(); st.session_state.update(gd=[e._demand() for _ in range(10)], gday=0, gstock=5, gprofit=0.0, glog=[])
        seq, gday = st.session_state["gd"], st.session_state["gday"]
        if gday < 10:
            st.write(f"**Day {gday+1} of 10** - stock this morning: **{st.session_state['gstock']}**")
            o = st.number_input("How many do you order?", 0, P["max_stock"], 7, key=f"g{gday}")
            if st.button("Open the shop", key=f"go{gday}"):
                pr, cr, i = day_calc(P, st.session_state["gstock"], o, seq[gday])
                st.session_state["gprofit"] += pr; st.session_state["gstock"] = cr
                st.session_state["glog"].append(i); st.session_state["gday"] += 1; st.rerun()
            if st.session_state["glog"]: st.dataframe(pd.DataFrame(st.session_state["glog"]), hide_index=True)
        else:
            s = 5; ap = 0.0
            for dmd in seq:
                o = min(int(policy[s]), P["max_stock"] - s); pr, s, _ = day_calc(P, s, o, dmd); ap += pr
            a, b = st.columns(2); you = st.session_state["gprofit"]
            a.metric("Your profit", f"AED {you:,.0f}"); b.metric("AI profit", f"AED {ap:,.0f}", f"{ap-you:+,.0f} vs you")
            st.dataframe(pd.DataFrame(st.session_state["glog"]), hide_index=True)
            if st.button("Play again"):
                for k in ["gd", "gday", "gstock", "gprofit", "glog"]: st.session_state.pop(k, None)
                st.rerun()
