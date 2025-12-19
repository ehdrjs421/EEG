import matplotlib.pyplot as plt

def compute_average_metrics(results):
    sens_list = [r['sensitivity'] for r in results if 'sensitivity' in r]
    spec_list = [r['specificity'] for r in results if 'specificity' in r]

    avg_sens = np.mean(sens_list) if sens_list else None
    avg_spec = np.mean(spec_list) if spec_list else None

    print(f"📊 평균 민감도 (Sensitivity): {avg_sens:.4f}")
    print(f"📊 평균 특이도 (Specificity): {avg_spec:.4f}")

    return avg_sens, avg_spec

# 평균 민감도 및 특이도 계산
avg_sens, avg_spec = compute_average_metrics(one_shot_results)

# 모델/계산 리소스 사용량
print("\n🧠 평균 모델/계산 리소스 사용량:")
display(df_results[['model_kb', 'sample_feature_kb', 'pred_time_s', 'testset_time_s']].mean())

# 전체 평균 Latency 출력
a = []
for i in range(len(df_results['latencies'])):
    latencies_for_patient = df_results['latencies'][i]
    # Filter out None values before calculating mean
    valid_latencies = [l for l in latencies_for_patient if l is not None]
    if valid_latencies:
        mean_latency = np.mean(valid_latencies)
        a.append(round(mean_latency, 2))
    else:
        a.append(np.nan) # Append NaN if no valid latencies

overall_mean_latency = np.nanmean(a) if a else np.nan
print(f"\n📏 전체 평균 latency: {overall_mean_latency:.2f} sec")

# Plotting mean latency per patient
b = list(range(1, len(a) + 1))

plt.figure(figsize=(12, 5))
plt.bar(b, a, color='skyblue')
plt.xlabel("Patient #")
plt.ylabel("Mean Latency (sec)")
plt.title("Mean Detection Latency per Patient")
plt.xticks(b)
plt.ylim(0, np.nanmax(a) + 1 if a else 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
save_path ='/content/drive/MyDrive/plot_latency.png'
plt.savefig(save_path, dpi=300)
plt.show()

# 리스트 → DataFrame
df_results = df_results.sort_values(by='patient')

patients = df_results['patient'].tolist()
sensitivities = df_results['sensitivity'].tolist()
specificities = df_results['specificity'].tolist()

# 🎯 1. 민감도 그래프
plt.figure(figsize=(14, 5))
plt.bar(x=np.arange(len(patients)), height=sensitivities, color='skyblue')
plt.xticks(np.arange(len(patients)), patients, rotation=45)
plt.ylim(0, 1.05)
plt.ylabel("Sensitivity")
plt.title("Sensitivity per Patient")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
save_path ='/content/drive/MyDrive/plot_sensitivity.png'
plt.savefig(save_path, dpi=300)
plt.show()

# 🎯 2. 특이도 그래프
plt.figure(figsize=(14, 5))
plt.bar(x=np.arange(len(patients)), height=specificities, color='lightgreen')
plt.xticks(np.arange(len(patients)), patients, rotation=45)
plt.ylim(0, 1.05)
plt.ylabel("Specificity")
plt.title("Specificity per Patient")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
save_path ='/content/drive/MyDrive/plot_specificity.png'
plt.savefig(save_path, dpi=300)
plt.show()

# Specificity difference plot
x_coords = np.arange(1, len(patients) + 1)
y1 = df_results_before['specificity']
y2 = df_results['specificity']

plt.figure(figsize=(12, 6))
plt.plot(x_coords, y1, marker='o', label='before', color='blue')
plt.plot(x_coords, y2, marker='s', label='after', color='orange')
plt.title('Specificity Difference (Before vs After Online Tuning)')
plt.xlabel('Patient Number')
plt.ylabel('Specificity')
plt.legend()
plt.grid(True)
plt.xticks(ticks=np.arange(1, len(patients) + 1))
save_path ='/content/drive/MyDrive/line_plot_specificity.png'
plt.savefig(save_path, dpi=300)
plt.show()

# Sensitivity difference plot
x_coords = np.arange(1, len(patients) + 1)
y1 = df_results_before['sensitivity']
y2 = df_results['sensitivity']

plt.figure(figsize=(12, 6))
plt.plot(x_coords, y1, marker='o', label='before', color='blue')
plt.plot(x_coords, y2, marker='s', label='after', color='orange')
plt.title('Sensitivity Difference (Before vs After Online Tuning)')
plt.xlabel('Patient Number')
plt.ylabel('Sensitivity')
plt.legend()
plt.grid(True)
plt.xticks(ticks=np.arange(1, len(patients) + 1))
save_path ='/content/drive/MyDrive/line_plot_sensitivity.png'
plt.savefig(save_path, dpi=300)
plt.show()
