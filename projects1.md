 How is the success rate of NFL kickers making 50 yard field goals in domes vs outdoor stadiums from 2023-2024? 

##Research Question and Dataset:
This project looks at whether NFL kickers are more successful on 50+ yard field goals in dome or outdoor stadiums from 2022 to 2024. The dataset includes play-by-play NFL data, and the main variables are kick distance, field goal result, and stadium type. Each row represents one field goal attempt. The purpose of this analysis is to compare how often kickers make long field goals in different environments and to see whether stadium conditions affect performance. This is important because kicking success can directly influence game outcomes, especially in close games. The data comes from official NFL play-by-play records and was filtered to only include field goals from 50 yards or farther. This makes the analysis focused on the specific question of whether long kicks are easier or harder depending on stadium type. This dataset is useful because it connects raw game information to a real sports question. It shows that distance is a major factor in success, and by separating dome/indoor and outdoor attempts, we can compare how different conditions may affect kicking performance. Overall, the data helps explain how environmental factors influence long-distance kicking in the NFL. 

 ##Conceptualized/Operationalized Variables and Important context:
   My  variable is_make vs miss
This first Visual Tracks the attempt result (1 for success, 0 for miss/blocked). This serves as your primary outcome variable, allowing you to calculate and compare overall conversion percentages (Makes/Total Attempts) across the two stadium types. Also the distance each field goal is being kicked. Two questions that need to be answered. What is the context surrounding this problem? What do we need to know to understand this issue? Why is this question relevant, and who would care about these findings? The context surrounding this problem is sucess for made vs missed field goal in both indoor and outdoor stadiums seeing stats on makes and misses during the 2022-2024 seasons. The question is relevant because it gives us accurate data on how kickers performed during the season and how far they could kick from distance wise. I used Api's from NFLreadpy. My variable for this visual is sucess_rate_pct this tracks the percentage of success rate from indoor stadiums compared to outdoor stadiums. Two questions to ask for this are What is the context surrounding this problem? What do we need to know to understand this issue? Why is this question relevant, and who would care about these findings? The context surrounding this problem is what is the success rate from indoor stadiums compared to outdoor. Nfl teams would care to see these findings because it shows and accurate percentage of field goals in indoor stadium compared to out door stadiums. 

Visuals:
```
pbp = nfl.load_pbp([2022, 2023, 2024])

long_fg = (
    pbp.filter(
        (pl.col("play_type") == "field_goal")
        & (pl.col("kick_distance") >= 50)
        & (pl.col("field_goal_result").is_in(["made", "missed"]))
    )
    .to_pandas()
    .copy()
)

long_fg["result"] = long_fg["field_goal_result"].map({"made": "Made", "missed": "Missed"})

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    data=long_fg,
    x="kick_distance",
    hue="result",
    bins=12,
    multiple="stack",
    palette={"Made": "#2ca02c", "Missed": "#d62728"},
    edgecolor="white",
    alpha=0.9,
    ax=ax,
    legend=True,
)

ax.set_title("Long Field Goal Attempts: Made vs Missed")
ax.set_xlabel("Kick Distance (Yards)")
ax.set_ylabel("Attempts")

plt.tight_layout()
plt.show()
```
Visuals:
<img width="784" height="484" alt="Image" src="https://github.com/user-attachments/assets/d892b674-86d8-4a3e-8167-80b48b1561cf" />
Visual 2:
<img width="784" height="484" alt="Image" src="https://github.com/user-attachments/assets/1c6f79fb-8702-4e85-8193-6898cd38b716" />

Ethics and Limitations:
This dataset gives a clear look at how field goal attempts change as distance increases and how those attempts are distributed between successful and unsuccessful kicks. The most noticeable pattern is that long field goals become much less frequent as the distance grows, which makes sense because they are more difficult and riskier. The data also shows that the farther the kick, the more likely it is to be missed, which highlights the importance of distance as a major factor in field goal success. What makes this dataset especially useful is that it allows us to compare different conditions, such as stadium type or weather-related factors, rather than looking at field goals as one single category. By analyzing these variables, we can see how context influences performance. This helps move the project beyond simple counts and toward understanding why some kicks succeed while others fail. Overall, the dataset is valuable because it connects raw game data to a meaningful sports question: which conditions make long-range field goals more difficult or more successful. Some missing context to add onto this is that the dataset contains NFL play-by-play information from the 2022–2024 seasons, focusing specifically on field goal attempts from 50 yards or farther. It includes both successful and unsuccessful kicks, along with contextual variables such as stadium roof type and kick distance. By narrowing the analysis to long field goals, we can better understand how distance and environment affect kicking performance. This makes the dataset useful for evaluating not just whether a kick was made or missed, but also what conditions may influence that outcome.
Jupyter Notebook: [Project 1.html](ProjectDTSC-1.ipynb)
Dataset/API [nflreadpy](https://nflreadpy.nflverse.com)
No ai used on this project. Just watched youtube videos.
## Key Academic References
Pasteur, R. D., & Cunningham-Rhoads, K. (2014). An expectation-based metric for NFL field goal kickers. Journal of quantitative analysis in sports, 10(1), 49-66.
Pophale, A. (2025). A Statistical Analysis of The Effect of Game Pressure on Kicker Performance.
Ransome, K. S. (2024). NFL Rule Changes Favor Offenses; But Don't Defenses Win Championships?.
