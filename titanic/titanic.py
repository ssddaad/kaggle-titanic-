import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

RANDOM_STATE = 42
N_SPLITS = 5

# -----------------------------
# 1) Load
# -----------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
print(train.columns.tolist())
y = train["Survived"].astype(int)
full = pd.concat([train.drop(columns=["Survived"]), test], axis=0, ignore_index=True)

# -----------------------------
# 2) Feature Engineering helpers
# -----------------------------
def extract_title(name: str) -> str:
    # Title like Mr, Mrs, Miss, Master, Rare...
    t = re.search(r",\s*([^\.]+)\.", name)
    t = t.group(1).strip() if t else "None"
    mapping = {
        "Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Rare","Countess":"Rare","Dona":"Rare",
        "Sir":"Rare","Don":"Rare","Jonkheer":"Rare","Capt":"Rare","Col":"Rare","Major":"Rare","Rev":"Rare","Dr":"Rare"
    }
    return mapping.get(t, t)

def ticket_prefix(t: str) -> str:
    s = re.sub(r"[\d\.\/\s]", "", str(t)).upper()
    return s if s else "NONE"

def ticket_number(t: str) -> int:
    m = re.findall(r"\d+", str(t))
    return int(m[-1]) if m else -1

def cabin_deck(c: str) -> str:
    if pd.isna(c) or str(c).strip()=="":
        return "M"   # Missing
    return str(c).split()[0][0]  # first cabin's first letter

def cabin_count(c: str) -> int:
    if pd.isna(c) or str(c).strip()=="":
        return 0
    return len(str(c).split())

def surname(name: str) -> str:
    return str(name).split(",")[0].strip()

# -----------------------------
# 3) Build features
# -----------------------------
f = full.copy()

# Basic clean
f["Embarked"] = f["Embarked"].fillna("S")

# Titles / surname
f["Title"] = f["Name"].apply(extract_title)
f["Surname"] = f["Name"].apply(surname)
f["NameLen"] = f["Name"].str.len()

# Family features
f["FamilySize"] = f["SibSp"] + f["Parch"] + 1
f["IsAlone"] = (f["FamilySize"] == 1).astype(int)

# Cabin / Ticket
f["Deck"] = f["Cabin"].apply(cabin_deck)
f["CabinCount"] = f["Cabin"].apply(cabin_count)

f["TicketPrefix"] = f["Ticket"].apply(ticket_prefix)
f["TicketNumber"] = f["Ticket"].apply(ticket_number)
f["TicketGroupSize"] = f.groupby("Ticket")["Ticket"].transform("count")

# Fare impute by Pclass+Embarked
fare_group_median = f.groupby(["Pclass", "Embarked"])["Fare"].transform("median")
f["Fare"] = f["Fare"].fillna(fare_group_median)
f["Fare"] = f["Fare"].fillna(f["Fare"].median())
# Fare bins sometimes help trees
f["FareBin"] = pd.qcut(f["Fare"], 8, duplicates="drop").astype(str)

# Age impute by Title+Pclass+Sex median (strong + simple, avoids leakage)
age_group_median = f.groupby(["Title", "Pclass", "Sex"])["Age"].transform("median")
f["Age"] = f["Age"].fillna(age_group_median)
f["Age"] = f["Age"].fillna(f["Age"].median())

# Interactions
f["Pclass*Sex"] = f["Pclass"].astype(str) + "_" + f["Sex"]
f["FamilySize*Title"] = f["FamilySize"].astype(str) + "_" + f["Title"]

# -----------------------------
# 4) Target encoding (leak‑safe, OOF) for high‑card cols
# -----------------------------
def kfold_target_encode(train_df, test_df, col, target, n_splits=5, smoothing=20):
    """
    Leak-safe KFold target encoding.
    train_df/test_df: 仅特征，不包含目标列
    target: 训练集目标 Series
    """
    prior = target.mean()
    oof = pd.Series(np.nan, index=train_df.index, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for tr_idx, va_idx in skf.split(train_df, target):
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        y_tr = target.iloc[tr_idx]

        tmp = pd.DataFrame({col: tr[col].astype(str).values, "_y": y_tr.values})
        counts = tmp[col].value_counts()
        means = tmp.groupby(col)["_y"].mean()
        enc = (means * counts + prior * smoothing) / (counts + smoothing)

        oof.iloc[va_idx] = va[col].map(enc).fillna(prior)

    tmp_full = pd.DataFrame({col: train_df[col].astype(str).values, "_y": target.values})
    counts_full = tmp_full[col].value_counts()
    means_full = tmp_full.groupby(col)["_y"].mean()
    enc_full = (means_full * counts_full + prior * smoothing) / (counts_full + smoothing)

    test_encoded = test_df[col].map(enc_full).fillna(prior)
    return oof, test_encoded


train_part = f.iloc[: len(train)]
test_part  = f.iloc[len(train):].reset_index(drop=True)

# Columns to target-encode
high_card_cols = ["Surname", "Ticket", "TicketPrefix"]
for c in high_card_cols:
    tr_enc, te_enc = kfold_target_encode(train_part, test_part, c, y, n_splits=N_SPLITS, smoothing=30)
    f.loc[: len(train)-1, c + "_TE"] = tr_enc.values
    f.loc[len(train):, c + "_TE"] = te_enc.values

# -----------------------------
# 5) Encode categorical for tree libs
# -----------------------------
cat_cols = [
    "Sex", "Embarked", "Title", "Deck", "FareBin",
    "Pclass*Sex", "FamilySize*Title", "TicketPrefix"
]
# Keep TE columns as numeric; for CatBoost we can also pass the raw categorical indices.
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    f[c] = le.fit_transform(f[c].astype(str))
    label_encoders[c] = le

# Drop text-heavy originals we replaced with encodings
drop_cols = ["Name", "Cabin", "Ticket", "Surname"]  # keep Ticket TE already
f = f.drop(columns=drop_cols)

# -----------------------------
# 6) Split back
# -----------------------------
X = f.iloc[: len(train)].copy()
X_test = f.iloc[len(train):].copy()
X["Survived"] = y.values  # for CatBoost Pool convenience later

features = [c for c in X.columns if c not in ["PassengerId", "Survived"]]

# -----------------------------
# 7) CV training per model
# -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
pred_lgb = np.zeros(len(test))
pred_xgb = np.zeros(len(test))
pred_cat = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X[features], y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31, max_depth=-1,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, min_child_samples=15,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    lgb_clf.fit(
        X_tr[features], y_tr,
        eval_set=[(X_va[features], y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )
    oof_lgb[va_idx] = lgb_clf.predict_proba(X_va[features])[:, 1]
    pred_lgb += lgb_clf.predict_proba(X_test[features])[:, 1] / N_SPLITS

    # XGBoost
    xgb_clf = XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, reg_alpha=0.0, min_child_weight=1.0, objective="binary:logistic",
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"
    )
    xgb_clf.fit(
        X_tr[features], y_tr,
        eval_set=[(X_va[features], y_va)],
        verbose=False,

    )
    oof_xgb[va_idx] = xgb_clf.predict_proba(X_va[features])[:, 1]
    pred_xgb += xgb_clf.predict_proba(X_test[features])[:, 1] / N_SPLITS

    # CatBoost (handles categorical patterns well even after label encoding)
    cat = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=5, l2_leaf_reg=3.0,
        random_state=RANDOM_STATE, loss_function="Logloss", eval_metric="Logloss",
        verbose=False
    )
    pool_tr = Pool(X_tr[features], y_tr)
    pool_va = Pool(X_va[features], y_va)
    cat.fit(pool_tr, eval_set=pool_va, use_best_model=True, early_stopping_rounds=300)
    oof_cat[va_idx] = cat.predict_proba(X_va[features])[:, 1]
    pred_cat += cat.predict_proba(X_test[features])[:, 1] / N_SPLITS

    acc_lgb = accuracy_score(y_va, (oof_lgb[va_idx] >= 0.5).astype(int))
    acc_xgb = accuracy_score(y_va, (oof_xgb[va_idx] >= 0.5).astype(int))
    acc_cat = accuracy_score(y_va, (oof_cat[va_idx] >= 0.5).astype(int))
    print(f"Fold {fold} acc - LGB: {acc_lgb:.4f} | XGB: {acc_xgb:.4f} | CAT: {acc_cat:.4f}")

# -----------------------------
# 8) Blending + OOF score
# -----------------------------
# weights tuned for robust LB
w_lgb, w_xgb, w_cat = 0.5, 0.3, 0.2
oof_blend = w_lgb*oof_lgb + w_xgb*oof_xgb + w_cat*oof_cat
cv_acc = accuracy_score(y, (oof_blend >= 0.5).astype(int))
print(f"CV accuracy (OOF, blended): {cv_acc:.5f}")

pred_blend = w_lgb*pred_lgb + w_xgb*pred_xgb + w_cat*pred_cat
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": (pred_blend >= 0.5).astype(int)
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
