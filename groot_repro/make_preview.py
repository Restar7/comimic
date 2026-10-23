"""Contact sheet: successful closed-loop rollouts (carton, can)."""
import cv2, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLIPS = [
    (sorted(glob.glob("videos/carton_success/*.mp4"))[0], "carton -> box  (success)"),
    (sorted(glob.glob("videos/can_success/*.mp4"))[0], "can on floor -> platform  (success)"),
]
frames = [0, 200, 400, 599]
fig, axes = plt.subplots(2, 4, figsize=(14, 5.6))
for r, (path, title) in enumerate(CLIPS):
    cap = cv2.VideoCapture(path)
    for c, idx in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        ax = axes[r, c]
        if ok:
            ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"t={idx/10:.0f}s", fontsize=9); ax.axis("off")
        if c == 0:
            ax.text(-0.06, 0.5, title, transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontsize=10, fontweight="bold")
    cap.release()
fig.suptitle("Closed-loop rollouts from the evaluation set (policy -> WBC -> MuJoCo), 640x480", fontsize=11)
fig.tight_layout(rect=(0.04, 0, 1, 0.94))
fig.savefig("out/rollout_preview.png", dpi=130)
print("wrote out/rollout_preview.png")
