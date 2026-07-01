"""Pipeline machine learning untuk klasifikasi credit score (preprocessing, training, evaluasi).
Berbasis OOP agar mudah digunakan ulang pada proses training di AWS SageMaker."""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Target tiga kelas, diberi kode angka untuk pelatihan
LABEL_MAP = {"Poor": 0, "Standard": 1, "Good": 2}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Nilai sampah/placeholder yang harus diperlakukan sebagai data kosong
GARBAGE = {"_______", "_", "!@9#%8", "", "nan", "NM", "#F%$D@*&8"}
# Kolom numerik yang tersimpan sebagai teks (mengandung underscore)
DIRTY_NUMERIC = ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
                 "Changed_Credit_Limit", "Outstanding_Debt",
                 "Amount_invested_monthly", "Monthly_Balance"]
# Kolom identitas yang dibuang agar tidak menyebabkan kebocoran data
DROP_COLS = ["Unnamed: 0", "ID", "Name", "SSN"]


class DataPreprocessor:
    """Membersihkan data mentah dan membangun transformer (imputasi, scaling, encoding)."""

    def __init__(self):
        self.transformer = None
        self.num_cols = None
        self.cat_cols = None

    @staticmethod
    def _to_number(series):
        return pd.to_numeric(
            series.astype(str).str.replace("_", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .replace({"": np.nan, "nan": np.nan}), errors="coerce")

    @staticmethod
    def _parse_history(val):
        # Ubah "X Years and Y Months" menjadi total bulan
        if pd.isna(val):
            return np.nan
        y = re.search(r"(\d+)\s*Year", str(val))
        m = re.search(r"(\d+)\s*Month", str(val))
        return (int(y.group(1)) * 12 if y else 0) + (int(m.group(1)) if m else 0)

    @staticmethod
    def _count_loans(val):
        # Hitung jumlah jenis pinjaman (satu nasabah bisa lebih dari satu)
        if pd.isna(val) or str(val).strip() == "":
            return 0
        parts = re.split(r",| and ", str(val))
        return len([p for p in parts if p.strip() and p.strip() != "Not Specified"])

    def clean_raw(self, df):
        """Pembersihan per baris: buang sampah, koreksi nilai mustahil, transformasi kolom khusus."""
        df = df.copy()
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
        for c in DIRTY_NUMERIC:
            if c in df.columns:
                df[c] = self._to_number(df[c])
        # Nilai di luar rentang wajar diubah menjadi kosong untuk diimputasi
        bounds = {"Age": (14, 100), "Num_Bank_Accounts": (0, 50), "Num_Credit_Card": (0, 50),
                  "Interest_Rate": (0, 50), "Num_of_Loan": (0, 50),
                  "Num_of_Delayed_Payment": (0, 100), "Num_Credit_Inquiries": (0, 50)}
        for c, (lo, hi) in bounds.items():
            if c in df.columns:
                df[c] = self._to_number(df[c]) if df[c].dtype == object else df[c]
                df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
        for c in ["Occupation", "Credit_Mix", "Payment_Behaviour"]:
            if c in df.columns:
                df[c] = df[c].replace(list(GARBAGE), np.nan)
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age"] = df["Credit_History_Age"].apply(self._parse_history)
        if "Type_of_Loan" in df.columns:
            df["Num_Loan_Types"] = df["Type_of_Loan"].apply(self._count_loans)
            df.drop(columns=["Type_of_Loan"], inplace=True)
        return df

    def impute_by_customer(self, df):
        """Mengisi nilai kosong pada kolom yang konsisten per nasabah."""
        df = df.copy()
        for c in ["Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary"]:
            if c in df.columns:
                df[c] = df.groupby("Customer_ID")[c].transform(
                    lambda s: s.fillna(s.mode().iloc[0]) if s.notna().any() else s)
        return df

    def build_transformer(self, X):
        """Imputasi median + scaling untuk numerik, modus + one-hot untuk kategorikal."""
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [c for c in X.columns if c not in self.num_cols]
        self.transformer = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), self.num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), self.cat_cols),
        ])
        return self.transformer

    def prepare_training_frame(self, df_raw):
        """Menghasilkan fitur (X), target (y), dan grup nasabah untuk validasi."""
        y = df_raw["Credit_Score"].map(LABEL_MAP)
        df = self.clean_raw(df_raw.drop(columns=["Credit_Score"]))
        df = self.impute_by_customer(df)
        groups = df["Customer_ID"].values
        X = df.drop(columns=[c for c in ["Customer_ID", "Month"] if c in df.columns])
        return X, y, groups


def model_zoo():
    """Kumpulan model yang dibandingkan dalam eksperimen."""
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9,
                            objective="multi:softprob", num_class=3,
                            eval_metric="mlogloss", random_state=42, n_jobs=-1)
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        xgb = GradientBoostingClassifier(random_state=42)
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               n_jobs=-1, random_state=42),
        "XGBoost": xgb,
    }


class ModelTrainer:
    """Melatih model di atas transformer preprocessing."""

    def __init__(self, preprocessor):
        self.pre = preprocessor

    def train(self, X, y, estimator):
        transformer = self.pre.build_transformer(X)
        pipe = Pipeline([("pre", transformer), ("clf", estimator)])
        # Pembobotan kelas untuk menangani data tidak seimbang
        if estimator.__class__.__name__ == "XGBClassifier":
            sw = compute_sample_weight("balanced", y)
            pipe.fit(X, y, clf__sample_weight=sw)
        else:
            pipe.fit(X, y)
        return pipe


class ModelEvaluator:
    """Menghitung metrik evaluasi (utama: macro F1, bukan akurasi)."""

    @staticmethod
    def evaluate(pipe, X, y):
        pred = pipe.predict(X)
        return {"macro_f1": f1_score(y, pred, average="macro"),
                "accuracy": accuracy_score(y, pred),
                "report": classification_report(y, pred, target_names=list(LABEL_MAP)),
                "confusion_matrix": confusion_matrix(y, pred)}

    @staticmethod
    def cross_val_macro_f1(estimator, pre, X, y, groups, n_splits=3):
        # Validasi silang berbasis grup nasabah agar tidak terjadi kebocoran data
        from sklearn.base import clone
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, va in cv.split(X, y, groups):
            transformer = pre.build_transformer(X)
            pipe = Pipeline([("pre", transformer), ("clf", clone(estimator))])
            if estimator.__class__.__name__ == "XGBClassifier":
                sw = compute_sample_weight("balanced", y.iloc[tr])
                pipe.fit(X.iloc[tr], y.iloc[tr], clf__sample_weight=sw)
            else:
                pipe.fit(X.iloc[tr], y.iloc[tr])
            scores.append(f1_score(y.iloc[va], pipe.predict(X.iloc[va]), average="macro"))
        return float(np.mean(scores)), float(np.std(scores))
