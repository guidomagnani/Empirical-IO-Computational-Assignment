import pandas as pd
dir= "../docs/Python_codes/"   

# Helper: load the estimates + SE + f-value per run
def load_run(tag):
    df1 = pd.read_csv(f"{dir}theta1_{tag}.csv")
    theta1 = df1["Theta1"].tolist()
    se1    = df1["Std.Error_theta1"].tolist()
    
    df2 = pd.read_csv(f"{dir}theta2_{tag}.csv")
    theta2 = df2["Theta2"].tolist()
    se2    = df2["Std.Error_theta2"].tolist()
    
    # pad theta1 to length 20
    while len(theta1) < 20:
        theta1.append("")
        se1.append("")
    
    # pad theta2 to length 6
    while len(theta2) < 6:
        theta2.append("")
        se2.append("")
    
    fval = pd.read_csv(f"{dir}fval_{tag}.csv").iloc[0,0]
    
    return theta1, se1, theta2, se2, fval


# Load all four runs
runs = ["one", "two", "three", "four"]
theta1_list, se1_list, theta2_list, se2_list, fvals = [], [], [], [], []

for r in runs:
    t1, s1, t2, s2, fv = load_run(r)
    theta1_list.append(t1)
    se1_list.append(s1)
    theta2_list.append(t2)
    se2_list.append(s2)
    fvals.append(fv)

# ---- BUILD TABLE OF STRINGS ----
table = pd.DataFrame("", index=range(27), columns=["Run1", "Run2", "Run3", "Run4"])

def fmt(est, se):
    """Return estimate and SE rounded to 3 decimals."""
    if est == "" or se == "":
        return ""
    return f"{est:.3f}\n({se:.3f})"


#Rows 1–20: theta1
for col in range(4):
    for i in range(20):
        table.iloc[i, col] = fmt(theta1_list[col][i], se1_list[col][i])

# Rows 21–26: theta2
for col in range(4):
    for i in range(6):
        table.iloc[20+i, col] = fmt(theta2_list[col][i], se2_list[col][i])

# Row 27: f-values
for col in range(4):
    table.iloc[26, col] = f"{fvals[col]:.6g}"

# Save final table to directory
table.to_csv(f"{dir}BLP_combined_results_latex_ready.csv", index=False)


# ---- CONVERT TO LATEX TABLE ----

df = pd.read_csv(f"{dir}BLP_combined_results_latex_ready.csv")

def to_latex_cell(x):
    if pd.isna(x) or x == "":
        return ""
    lines = x.split("\n")
    if len(lines) == 1:
        return lines[0]
    est = lines[0]
    se  = lines[1]
    return f"\\shortstack{{{est}\\\\ {se}}}"

latex_df = df.applymap(to_latex_cell)

latex_str = "\\begin{table}[ht]\n\\centering\n"
latex_str += "\\begin{tabular}{lcccc}\n"
latex_str += "\\hline\n"
latex_str += "Parameter & Run1 & Run2 & Run3 & Run4 \\\\\n"
latex_str += "\\hline\n"

# ---- Row Labels ----
for i in range(len(latex_df)):
    if i < 20:
        # theta_1,1 ... theta_1,20
        label = f"$\\theta_{{1,{i+1}}}$"
    elif i < 26:
        # theta_2,1 ... theta_2,6
        label = f"$\\theta_{{2,{i-19}}}$"
    else:
        label = "fval"
    
    row = latex_df.iloc[i]
    latex_str += f"{label} & {row['Run1']} & {row['Run2']} & {row['Run3']} & {row['Run4']} \\\\\n"


latex_str += "\\hline\n\\end{tabular}\n"
latex_str += "\\caption{BLP Estimates Across Four Runs}\n"
latex_str += "\\end{table}\n"


with open("blp_results_table.tex", "w") as f:
    f.write(latex_str)

print("LaTeX table saved to blp_results_table.tex")
