# Question 
What factors best predict whether an NFL offense will successfully convert a fourth-down attempt during a game from 2020-2024?
# My target variable is nfl_data_py
# Probelem Definition 
This is a classification problem that benefits NFL coaching staff, front offices, roster constructors, broadcasters, analysts, and media. Investigating this problem is meaningful because fourth-down decisions are among the most impactful moments in an NFL game. A single failed conversion turns the ball over to the opponent with field position advantage, while a successful conversion extends a scoring drive; because these plays heavily shift a team’s Win Probability and Expected Points Added, optimizing fourth-down strategy directly influences wins and losses over a season.
# Background and Context
To understand the problem, the reader needs to know that an offense gets four attempted downs to gain 10 yards, and gaining 10 or more yards resets the count to a new 1st down. This approach is informed by research establishing that historical play-calling does not equal optimal play-calling, proving why data models are necessary to replace human bias with objective probabilities. Furthermore, credible sources suggest that variables such as yards to go, field position, score, and time remaining heavily matter when evaluating fourth-down decisions.
## Code:
```
seasons = [2020, 2021, 2022, 2023, 2024]
pbp = import_pbp_data(seasons)

df = pbp[(pbp["down"] == 4) & (pbp["play_type"].isin(["pass", "rush"]))].copy()

if "yards_to_go" not in df.columns:
    if "ydstogo" in df.columns:
        df["yards_to_go"] = df["ydstogo"]
    else:
        df["yards_to_go"] = 0

df["yards_to_go"] = df["yards_to_go"].fillna(0)
df["yards_gained"] = df["yards_gained"].fillna(0)
df["success"] = (df["yards_gained"] >= df["yards_to_go"]).astype(int)

df["goal_to_go"] = (df["yardline_100"] <= 10).astype(int)
df["red_zone"] = (df["yardline_100"] <= 20).astype(int)
df["score_diff"] = df["score_differential"].fillna(0)

X = df[["yards_to_go", "yardline_100", "goal_to_go", "red_zone", "score_diff", "play_type"]]
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), ["yards_to_go", "yardline_100", "goal_to_go", "red_zone", "score_diff"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), ["play_type"])
    ]
)

model = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Failed", "Converted"],
            yticklabels=["Actual Failed", "Actual Converted"])
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("Baseline accuracy:", round(accuracy_score(y_test, baseline_pred), 3))

plt.title("4th Down Conversion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```
## Visual 1:
<img width="539" height="455" alt="JETS" src="https://github.com/user-attachments/assets/6719e550-7e7e-4abe-be4a-ac1625b11bf6" />

# Data Description
Data came from nfl_data_py. In the nfl_data_py dataset, each observation or row represents an individual play recorded on 4th down where play_type is classified as a pass (pass = 1) or a rush (rush = 1). The dataset starts at about 250,000 total plays in pbp, while the filtered fourth-down dataset (df) contains roughly 3,000 plays. The target variable is success, a binary classification target where 1 means the play gained at least the yards needed for a first down (converted), and 0 means it did not (failed). Potential features are available to evaluate these decisions, though data collection was subject to assumptions, restrictions, and limitations: the data is limited to recorded NFL play-by-play from the 2020–2024 seasons, strictly including only 4th down plays and how they turned out.
# Code 2: 
```
counts = df["success"].value_counts().reindex([0, 1], fill_value=0)
labels = ["Failed", "Converted"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, counts.values, color=["#d96b6b", "#4c9f70"])

ax.set_title("Fourth-Down Play Outcomes")
ax.set_xlabel("Outcome")
ax.set_ylabel("Number of Plays")

total = counts.sum()
for bar, count in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,} ({count / total:.1%})",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()
```
## Visual 2: 
<img width="690" height="490" alt="Image" src="https://github.com/user-attachments/assets/02972368-bee0-4157-9c68-5e4e473234e1" />

# Data Understanding and Exploration
Summary statistics reveal the typical range of yards needed, yards gained, field position, and score difference for the fourth-down plays analyzed. The target variable's distribution is shown on an outcome chart displaying counts and percentages, describing the classes as imbalanced if one outcome is noticeably more common than the other, or relatively balanced otherwise. In terms of patterns, relationships, unusual values, or outliers, the code checks conversion rates by yards to go and play type while flagging potential outliers in yards to go, yards gained, and field position. Key visualizations that aid in understanding these variables include the outcome bar chart showing the balance between failed and converted plays, the conversion rate table revealing how distance and play call relate to success, and the confusion matrix detailing where the model’s predictions are correct or mistaken. Finally, this data exploration directly informed feature-selection and preprocessing decisions by identifying yards to go, field position, and play type as core features, encoding play type, scaling numeric features to fit the model, and flagging unusual values or missing yardage for further evaluation.
# Data Preparation and Feature Selection
To prepare the data, missing values were not messed with while rows missing yards to go or yards gained were excluded prior to target definition; duplicate observations and outliers were not removed or separately checked. For feature selection, yards to go, field position goal to go, red-zone status, score difference, and play type were included because they describe the situation prior to the fourth-down attempt (noting goal-to-go and red-zone flags overlap with yardline_100), whereas yards gained was excluded to avoid data leakage since it defines the target. Categorical and numerical variables were transformed by one-hot encoding play_type and scaling. Data was separated for training and evaluation. Finally, data leakage was prevented by excluding yards_gained from the predictors and placing imputation, scaling, and encoding inside a pipeline so transformations were fit solely on training data, though the random split mixes seasons rather than evaluating prediction performance on future seasons.
# Baseline and Model Development
To establish a benchmark, I used a most-frequent-class baseline that always predicts the more common outcome, which gives a better idea than just guessing. I trained a logistic regression model and compared it with this most-frequent-class dummy classifier as the baseline. These models were appropriate for the prediction problem because the target has two outcomes converted or failed—and the dummy classifier provides a simple benchmark to verify whether logistic regression performs better than always predicting the majority class. I did not tune any model settings or hyperparameters. Finally, I ensured the models were compared fairly by testing both on the exact same training and test split, evaluating the baseline and logistic regression on the identical test set so their accuracy scores could be compared directly.
#  Model Evaluation and Selection
To evaluate performance, I used accuracy, precision, recall, to measure how often the model is correct and how well it identifies both successful conversions and failed attempts. Model one performed better because it did not limit all the data compared to the second model. Ultimately, I selected logistic regression as my final model because it effectively predicts the two outcomes—converted or failed—using the play situation, a decision supported by comparing converted versus non-converted predictions. In terms of tradeoffs between metrics, accuracy gives a useful overall score but can hide mistakes on less common outcomes, whereas ROC AUC better illustrates how well the model distinguishes conversions from failures, though it should be noted that logistic regression was evaluated against a baseline rather than a second trained model.
# Model Interpretation and Insights
The model uses yards to go, field position, goal-to-go, red-zone status, score difference, and play type to estimate the chance of a conversion, though determining which features are most influential would require examining its coefficients directly. To see where the model performs well or poorly. From these results, we can conclude how situational factors like yards to go, field position, score difference, and play type relate to fourth-down conversions in the 2020–2024 NFL data. However, the model cannot prove these factors cause success or reliably predict every play, as it only evaluates pass and rush attempts, and imputing missing yardage with zero may impact overall performance.
# Limitations, Ethics, and Reflection
Several opinions exist in the dataset because it only includes fourth-down pass and rush attempts from 2020–2024, excluding punts and field goals. Coaches and teams could be directly affected by incorrect predictions, where a false positive predicts a conversion that ultimately fails and a false negative predicts a failure when the team would have actually converted—either error could lead to poor fourth-down strategy and affect the game's outcome. Despite these risks, the model would be appropriate for real-world decision-making because leveraging this data can help teams make strategic choices to win games. To build on this foundation, future work would incorporate game context such as time remaining, score, and opponent defense, compare logistic regression against tree-based models, and evaluate performance on a later season. Ultimately, users must understand that the model's predictions are estimates rather than guarantees, given its reliance on a limited feature set and historical 2020–2024 pass and rush attempts.

# Code and AI transparency:
Jupyter Notebook: [Project2.html](Project2222-1.ipynb)

Dataset/API: [NFL-data-py](https://pypi.org/project/nfl-data-py/)

No ai was used on this project I used youtube to help me out. 
# Key Academic References: 
Mecha, L. (2026). xScore: A Machine Learning Framework for Evaluating NFL Team Performance and Playoff Success. Available at SSRN 6870618.

Raymond, S. (2025). Fourth downs decoded: a predictive and causal analysis of player impact and decision making in the NFL.

Yam, D. R., & Lopez, M. J. (2019). What was lost? A causal estimate of fourth down behavior in the National Football League. Journal of Sports Analytics, 5(3), 153-167.
