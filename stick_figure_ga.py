import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave

# ----------------------------
# 1) Parameters & Beat Pattern
# ----------------------------
bpm = 60
mean_period = 60.0 / bpm
fast = mean_period * 0.5
slow = mean_period * 1.5
# pattern = [fast]*4 + [slow]*3 + [fast]*4 + [slow]*2 + [fast]*4 + [slow]*2
pattern = (
    [slow]*8             # → Intro: 8 slow beats (gentle wave build-up)
  + [fast]*4 + [slow]*4  # → Verse: 4 fast (twerk), 4 slow (wave)
  + [slow]*2 + [fast]*6  # → Pre-chorus: 2 slow, 6 fast crescendo
  + [slow]*1 + [fast]*8 + [slow]*2  # → Chorus: 1 slow, 8 fast, 2 slow
  + [slow]*4             # → Bridge break: 4 slow
  + [fast]*12            # → Drop: 12 fast (non-stop twerk)
)
beat_times = np.cumsum(pattern)
total_time = beat_times[-1] + mean_period
deltas = pattern  # each is the actual interval length
# Anything shorter than (mean_period * 0.75) we’ll treat as “fast”
thresh = mean_period * 0.75
beat_types = np.array([dt < thresh for dt in deltas])
# beat_types = np.array([interval == fast for interval in pattern])
print(beat_times)
print(beat_types)
M = len(beat_times)
fps = 30

# ----------------------------
# 2) GA Parameters
# ----------------------------
population_size = 100
generations = 150
mutation_rate = 0.1

# ----------------------------
# 3) Stick Figure Geometry
# ----------------------------
hip_base_y    = 1.0
hip_x         = 0.4
foot1         = (0.45, 0.0)
foot2         = (0.55, 0.0)
head_offset   = (0.7, 1.5)
amplitude     = 0.15
shoulder_frac = 0.3
bend_strength = 0.2
knee_frac     = 0.5

eye_dx, eye_dy   = 0.08, 0.1
mouth_w, mouth_dy = 0.2, 0.1

# ----------------------------
# 4) Generate metronome WAV
# ----------------------------
sample_rate = 44100
t_audio = np.linspace(0, total_time, int(sample_rate * total_time), endpoint=False)
metronome = np.zeros_like(t_audio)
click_dur = 0.01
click_samps = int(click_dur * sample_rate)
freq = 1000
t_click = np.linspace(0, click_dur, click_samps, endpoint=False)
click_wave = 0.5 * np.sin(2 * np.pi * freq * t_click)
for bt in beat_times:
    idx = int(bt * sample_rate)
    metronome[idx:idx+click_samps] += click_wave

wav_file = 'metronome.wav'
with wave.open(wav_file, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes((metronome * 32767).astype(np.int16).tobytes())

try:
    import sounddevice as sd
    sd.play(metronome.astype(np.float32), samplerate=sample_rate)
except ImportError:
    pass  # no audio playback

# ----------------------------
# 5) Genetic Algorithm
# ----------------------------
def fitness(phi):
    return np.sum(np.sin(2*np.pi*(bpm/60.0)*beat_times + phi))

pop = np.random.uniform(0, 2*np.pi, (population_size, M))
best_phis = []
fitness_history = []

for gen in range(generations):
    scores = np.array([fitness(ind) for ind in pop])
    best = np.argmax(scores)
    best_phis.append(pop[best])
    fitness_history.append(scores[best])
    winners = pop[np.argsort(scores)[-population_size//2:]]
    children = []
    while len(children) < population_size:
        p1, p2 = winners[np.random.choice(len(winners), 2, replace=False)]
        child = (p1 + p2) / 2 + np.random.normal(0, mutation_rate, size=M)
        children.append(np.mod(child, 2*np.pi))
    pop = np.array(children)

best_phis = np.array(best_phis)

# ----------------------------
# 6) Alignment Info
# ----------------------------
def alignment_error(phi):
    return np.mean(np.abs(1 - np.sin(2*np.pi*(bpm/60.0)*beat_times + phi)))

initial = np.random.uniform(0, 2*np.pi, M)
print("Beat times:", np.round(beat_times, 3))
print("Init fitness:", fitness(initial), "Final fitness:", fitness(best_phis[-1]))
print("Init error:", alignment_error(initial), "Final error:", alignment_error(best_phis[-1]))

# ----------------------------
# 7) Training Animation + Growth Curve
# ----------------------------
total_frames = int(total_time * fps)
frames_per_gen = total_frames / generations

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel setup
ax1.set_xlim(0, 1.5); ax1.set_ylim(0, 3); ax1.axis('off')
ax1.plot(*foot1, 'k_', ms=20); ax1.plot(*foot2, 'k_', ms=20)
th1, = ax1.plot([], [], lw=3); sh1, = ax1.plot([], [], lw=3)
th2, = ax1.plot([], [], lw=3); sh2, = ax1.plot([], [], lw=3)
back_line, = ax1.plot([], [], lw=3)
larm, = ax1.plot([], [], lw=2); rarm, = ax1.plot([], [], lw=2)
head = plt.Circle((0, 0), 0.3, fill=False, lw=2); ax1.add_patch(head)
e1, = ax1.plot([], [], 'o', ms=4); e2, = ax1.plot([], [], 'o', ms=4)
mouth, = ax1.plot([], [], lw=1)
k1dot, = ax1.plot([], [], 'o', ms=6); k2dot, = ax1.plot([], [], 'o', ms=6)
beat_dot, = ax1.plot([], [], 'o', color='red', ms=8)
action_text = ax1.text(0.75, 2.8, '', fontsize=16, ha='center')
type_text   = ax1.text(0.75, 2.6, '', fontsize=14, ha='center', color='blue')

# Right panel setup
ax2.set_xlim(0, generations - 1)
ymin, ymax = min(fitness_history), max(fitness_history)
ax2.set_ylim(ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin))
ax2.set_xlabel('Generation'); ax2.set_ylabel('Max Fitness'); ax2.set_title('Learning Curve')
curve, = ax2.plot([], [], lw=2)
action_text = ax1.text(0.75, 2.8, '', fontsize=16, ha='center')
type_text   = ax1.text(0.75, 2.6, '', fontsize=14, ha='center', color='blue')

def init_train():
    for ln in [th1, sh1, th2, sh2, back_line, larm, rarm, e1, e2, mouth, k1dot, k2dot, beat_dot, curve]:
        ln.set_data([], [])
    head.set_center((0, 0))
    action_text.set_text('')
    type_text.set_text('') 
    return [th1, sh1, th2, sh2, back_line, larm, rarm, head, e1, e2, mouth, k1dot, k2dot, beat_dot, action_text, type_text, curve]

def animate_train(frame):
    t = frame / fps
    gen = min(generations - 1, int(frame / frames_per_gen))
    phi = best_phis[gen]
    idx = np.searchsorted(beat_times, t) - 1
    idx = np.clip(idx, 0, M-2)
    dt = pattern[idx]
    raw = np.sin(2*np.pi*(t - beat_times[idx])/dt + phi[idx])

    # Fast beat: twerk
    if beat_types[idx]:
        hy = hip_base_y + amplitude * raw
        hpt = (hip_x + head_offset[0], hy + head_offset[1])
        head.set_center(hpt)
        k1 = ((hip_x + foot1[0])*knee_frac, (hy + foot1[1])*knee_frac + bend_strength)
        k2 = ((hip_x + foot2[0])*knee_frac, (hy + foot2[1])*knee_frac + bend_strength)
        th1.set_data([hip_x, k1[0]], [hy, k1[1]]); sh1.set_data([k1[0], foot1[0]], [k1[1], foot1[1]])
        th2.set_data([hip_x, k2[0]], [hy, k2[1]]); sh2.set_data([k2[0], foot2[0]], [k2[1], foot2[1]])
        back_line.set_data([hip_x, hpt[0]], [hy, hpt[1]])
        sx = hip_x + (hpt[0] - hip_x)*shoulder_frac; sy = hy + (hpt[1] - hy)*shoulder_frac
        larm.set_data([sx, k1[0]], [sy, k1[1]]); rarm.set_data([sx, k2[0]], [sy, k2[1]])
        k1dot.set_data([k1[0]], [k1[1]]); k2dot.set_data([k2[0]], [k2[1]])
    # Slow beat: wave
    else:
        hy = hip_base_y
        hpt = (hip_x + head_offset[0], hy + head_offset[1])
        head.set_center(hpt)
        k1 = ((hip_x + foot1[0])*knee_frac, (hy + foot1[1])*knee_frac + bend_strength)
        k2 = ((hip_x + foot2[0])*knee_frac, (hy + foot2[1])*knee_frac + bend_strength)
        th1.set_data([hip_x, k1[0]], [hy, k1[1]]); sh1.set_data([k1[0], foot1[0]], [k1[1], foot1[1]])
        th2.set_data([hip_x, k2[0]], [hy, k2[1]]); sh2.set_data([k2[0], foot2[0]], [k2[1], foot2[1]])
        back_line.set_data([hip_x, hpt[0]], [hy, hpt[1]])
        sx = hip_x + (hpt[0] - hip_x)*shoulder_frac; sy = hy + (hpt[1] - hy)*shoulder_frac
        angle = -np.pi/2 + np.pi*(raw + 1)/2
        x2 = sx + np.cos(angle); y2 = sy + np.sin(angle)
        larm.set_data([sx, x2], [sy, y2]); rarm.set_data([sx, k2[0]], [sy, k2[1]])
        k1dot.set_data([], []); k2dot.set_data([], [])

    # Face
    e1.set_data([hpt[0] - eye_dx], [hpt[1] + eye_dy])
    e2.set_data([hpt[0] + eye_dx], [hpt[1] + eye_dy])
    mouth.set_data([hpt[0] - mouth_w/2, hpt[0] + mouth_w/2], [hpt[1] - mouth_dy]*2)

    # Beat indicator
    if np.any(np.isclose(t, beat_times, atol=1/fps)):
        beat_dot.set_data([1.3], [2.8])
    else:
        beat_dot.set_data([], [])

    # Determine action
    action = 'Twerk' if beat_types[idx] else 'Wave'
    action_text.set_text(action)
    type_text.set_text('Fast beat' if beat_types[idx] else 'Slow beat')  # ← update new label


    # Growth curve
    xs = np.arange(gen+1); ys = fitness_history[:gen+1]
    curve.set_data(xs, ys)

    return [th1, sh1, th2, sh2, back_line, larm, rarm, head, e1, e2, mouth, k1dot, k2dot, beat_dot, action_text, type_text, curve]

anim1 = animation.FuncAnimation(fig1, animate_train, init_func=init_train,
                                frames=total_frames, interval=1000/fps, blit=True)

# ----------------------------
# 8) Final-only Animation
# ----------------------------
fig2, axf = plt.subplots(figsize=(6,5))
axf.set_xlim(0,1.5); axf.set_ylim(0,3); axf.axis('off')
axf.plot(*foot1,'k_',ms=20); axf.plot(*foot2,'k_',ms=20)
f_th1,=axf.plot([],[],lw=3); f_sh1,=axf.plot([],[],lw=3)
f_th2,=axf.plot([],[],lw=3); f_sh2,=axf.plot([],[],lw=3)
f_back,=axf.plot([],[],lw=3)
f_larm,=axf.plot([],[],lw=2); f_rarm,=axf.plot([],[],lw=2)
f_head=plt.Circle((0,0),0.3,fill=False,lw=2); axf.add_patch(f_head)
f_e1,=axf.plot([],[],'o',ms=4); f_e2,=axf.plot([],[],'o',ms=4)
f_mouth,=axf.plot([],[],lw=1)
f_k1,=axf.plot([],[],'o',ms=6); f_k2,=axf.plot([],[],'o',ms=6)
f_beat,=axf.plot([],[],'o',color='red',ms=8)

best = best_phis[-1]

def init_final():
    for ln in [f_th1,f_sh1,f_th2,f_sh2,f_back,f_larm,f_rarm,f_e1,f_e2,f_mouth,f_k1,f_k2,f_beat]:
        ln.set_data([],[])
    f_head.set_center((0,0))
    return [f_th1,f_sh1,f_th2,f_sh2,f_back,f_larm,f_rarm,f_head,f_e1,f_e2,f_mouth,f_k1,f_k2,f_beat]

def animate_final(frame):
    t = frame / fps
    idx = np.searchsorted(beat_times, t) - 1
    idx = np.clip(idx, 0, M-2)
    dt = pattern[idx]
    raw = np.sin(2*np.pi*(t - beat_times[idx])/dt + best[idx])

    if beat_types[idx]:
        hy = hip_base_y + amplitude*raw
        hpt = (hip_x + head_offset[0], hy + head_offset[1])
        f_head.set_center(hpt)
        k1 = ((hip_x + foot1[0])*knee_frac, (hy + foot1[1])*knee_frac + bend_strength)
        k2 = ((hip_x + foot2[0])*knee_frac, (hy + foot2[1])*knee_frac + bend_strength)
        f_th1.set_data([hip_x, k1[0]], [hy, k1[1]]); f_sh1.set_data([k1[0], foot1[0]], [k1[1], foot1[1]])
        f_th2.set_data([hip_x, k2[0]], [hy, k2[1]]); f_sh2.set_data([k2[0], foot2[0]], [k2[1], foot2[1]])
        f_back.set_data([hip_x, hpt[0]], [hy, hpt[1]])
        sx = hip_x + (hpt[0] - hip_x)*shoulder_frac; sy = hy + (hpt[1] - hy)*shoulder_frac
        f_larm.set_data([sx, k1[0]], [sy, k1[1]]); f_rarm.set_data([sx, k2[0]], [sy, k2[1]])
        f_k1.set_data([k1[0]], [k1[1]]); f_k2.set_data([k2[0]], [k2[1]])
    else:
        hy = hip_base_y
        hpt = (hip_x + head_offset[0], hy + head_offset[1])
        f_head.set_center(hpt)
        k1 = ((hip_x + foot1[0])*knee_frac, (hy + foot1[1])*knee_frac + bend_strength)
        k2 = ((hip_x + foot2[0])*knee_frac, (hy + foot2[1])*knee_frac + bend_strength)
        f_th1.set_data([hip_x, k1[0]], [hy, k1[1]]); f_sh1.set_data([k1[0], foot1[0]], [k1[1], foot1[1]])
        f_th2.set_data([hip_x, k2[0]], [hy, k2[1]]); f_sh2.set_data([k2[0], foot2[0]], [k2[1], foot2[1]])
        f_back.set_data([hip_x, hpt[0]], [hy, hpt[1]])
        sx = hip_x + (hpt[0] - hip_x)*shoulder_frac; sy = hy + (hpt[1] - hy)*shoulder_frac
        angle = -np.pi/2 + np.pi*(raw + 1)/2
        x2 = sx + np.cos(angle); y2 = sy + np.sin(angle)
        f_larm.set_data([sx, x2], [sy, y2]); f_rarm.set_data([sx, k2[0]], [sy, k2[1]])
        f_k1.set_data([],[]); f_k2.set_data([],[])
    
    # Determine action
    action = 'Twerk' if beat_types[idx] else 'Wave'
    action_text.set_text(action)

    # face
    f_e1.set_data([hpt[0] - eye_dx], [hpt[1] + eye_dy])
    f_e2.set_data([hpt[0] + eye_dx], [hpt[1] + eye_dy])
    f_mouth.set_data([hpt[0] - mouth_w/2, hpt[0] + mouth_w/2], [hpt[1] - mouth_dy]*2)

    # beat indicator
    if np.any(np.isclose(t, beat_times, atol=1/fps)):
        f_beat.set_data([1.3], [2.8])
    else:
        f_beat.set_data([], [])

    return [f_th1,f_sh1,f_th2,f_sh2,f_back,f_larm,f_rarm,f_head,f_e1,f_e2,f_mouth,f_k1,f_k2,f_beat, action_text]

anim2 = animation.FuncAnimation(fig2, animate_final, init_func=init_final,
                                frames=total_frames, interval=1000/fps, blit=True)

plt.tight_layout()
plt.show()


# Time vector
t = np.linspace(0, total_time, 2000)

# Piecewise sine function
# def piecewise_sine(t, beat_times, phi):
#     idx = np.clip(np.searchsorted(beat_times, t) - 1, 0, len(beat_times)-2)
#     dt = beat_times[idx+1] - beat_times[idx]
#     x = t - beat_times[idx]
#     return np.sin(2 * np.pi * (bpm/60.0) * x / dt + phi[idx])

# y_initial = piecewise_sine(t, beat_times, best_phis[0])
# y_final   = piecewise_sine(t, beat_times, best_phis[-1])

# --------------------------------------
# OPTION A: single‐global‐phase sine

# choose a single phase (e.g. the mean of your learned per-beat phases)
phi_init_global  = np.mean(best_phis[0])
phi_final_global = np.mean(best_phis[-1])

# build two constant‐amplitude sines
y_initial = np.sin(2*np.pi*(bpm/60.0)*t + phi_init_global)
y_final   = np.sin(2*np.pi*(bpm/60.0)*t + phi_final_global)

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# 1) Learning curve placeholder
axes[0].plot(fitness_history, marker='o')
axes[0].set_title('Fitness Growth Over Generations')
axes[0].set_xlabel('Generation')
axes[0].set_ylabel('Max Fitness')

# 2) Initial sine waveform
axes[1].plot(t, y_initial, label='Initial')
# axes[1].scatter(beat_times, np.sin(2*np.pi*(bpm/60.0)*(beat_times-beat_times) + best_phis[0]),
#                 color='red', zorder=5)
axes[1].scatter(beat_times,
                np.sin(2*np.pi*(bpm/60.0)*beat_times + phi_init_global),
                color='red')
axes[1].set_title('Initial Piecewise Sine Alignment')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')

# 3) Final sine waveform
axes[2].plot(t, y_final, label='Final')
axes[2].scatter(beat_times, np.sin(2*np.pi*(bpm/60.0)*(beat_times-beat_times) + best_phis[-1]),
                color='green', zorder=5)
axes[2].set_title('Final Piecewise Sine Alignment')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()