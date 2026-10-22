# Project Q&A: InstantSim and Eddy3D

How to use: read a question, close the file, and answer out loud. Then open it
and compare. The answers are written the way you would say them in an
interview: first person, short sentences, conclusion first. Don't memorise the
wording; use your own. The code is there so you know what each sentence refers
to.

In Part 1, every question has:
- **Say:** what you say first, about 30 to 60 seconds
- the code it refers to
- **Follow-ups:** what the interviewer is likely to ask next, and how you'd answer

Code comes from the InstantSim repository: `training/train_custom_solar.py`
(model, data, training), `backend/app/models/networks.py` (the model used for
inference and evaluation), `training/evaluate_model.py` and
`training/probe_symmetry.py` (evaluation), `backend/app/core/inference.py` and
`backend/frontend/index.html` (the running tool).

---

## Part 0: Facts to know (checked 2026-10-21)

These are the numbers and facts to use. README and docs match them.

**Results** (200 held-out test scenes, `training/evaluate_model.py`)
- 90.8% of pixels within 1 hour of Ladybug. Mean error 0.56 hours.
- 59.3% exactly right.
- No-shadow baseline (ignore buildings, every pixel gets full sun): 53.6%
  within 1 hour, mean error 2.18 hours.
- On 49 training scenes: 95.4% within 1 hour, 70.3% exact, mean error 0.38 h.
- Near shadow edges: 49% exact. Away from edges: 91% exact.
- High-rise scenes: mean error 0.73 h. Low- and mid-rise: 0.40 h.
  Roofs: 0.92 h. Ground: 0.50 h.
- Don't use 80% any more. That was an early rough estimate (comparing images
  with ChatGPT). Don't use 57.1% / 90.9% / 0.58 h either; those were before a
  bug fix.

**The test set**
- The 200 test scenes have new building layouts, but reuse the site outlines
  and sun conditions of training scenes (199 of 200). So the numbers show the
  model handles buildings it hasn't seen, not new sites or dates.
- Mix: 51 low-rise (tallest about 50 m), 49 mid-rise, 100 high-rise (up to
  250 m).

**Data**
- 4,000 training scenes, 200 test scenes, made in Grasshopper.
- Site: a 1000 m × 1000 m square, scaled around its centre by 0.6–1.25 in x and
  0.6–1.3 in y. So sites are 600–1250 m by 600–1300 m.
- Buildings: the site is randomly split into plots and each plot is extruded.
  Height, density and footprint shape (polygon, rectangle, circle, L-shape…)
  are sampled by category. Say "parametric random generation", not
  "procedural".
- Location: latitude random between 35°S and 65°N, longitude anywhere. Date
  random over the whole year.
- Label: Ladybug direct sun hours, 9:00 to 17:00 local solar time. Grid 5 m,
  test points 0.2 m above the surface, one sun position per hour. 10 classes,
  0 to 9 hours. ("Nine classes" in the letters is wrong.)
- Ladybug takes roughly tens of seconds to a minute per scene.

**Height map**
- A 512×512 screenshot of the Rhino Top view, not a raycast. Training images
  were exported from the view; the live tool captures it with a Python script.
  Both give the same encoding.
- Ground 240. Background 255. Buildings: height 0–250 m remapped to 60–255,
  then a white-to-black gradient, so 0 m is gray 195 and 250 m is 0. Above
  250 m is clipped.
- Why 60: it leaves a gap between the lowest building (195) and the ground
  (240), so a very low building is never confused with ground.
- Rendered mode adds a gamma curve (about 1/2.2), so the saved pixel values are
  not linear in height: 0 m shows as about 226, 50 m as 204, 125 m as 165,
  250 m as 0.
- The camera zooms to 1.2 × the longer side of the site. So one pixel is
  1.4–3.0 m in the 512 px image and 2.8–6.1 m at the model's 256 px. The
  model is never told which.

**Model and training**
- Conditional U-Net generator (pix2pix-based), 60.1 million parameters.
- FiLM in 13 blocks: enc2–enc7 and dec1–dec7. enc1 and enc8 have no
  normalisation and no FiLM.
- InstanceNorm only, no BatchNorm.
- Loss: least-squares GAN + 50 × L1. No ablation was run for either choice.
- Adam, learning rate 1e-4, betas (0.5, 0.999), batch 16, 200 epochs.
- No data augmentation. No validation set.
- Checkpoints every 5 epochs; the released model is epoch 195 (counting from 0),
  because the script only saved when `epoch % 5 == 0`. Now it also saves at the
  end.
- Trained on your own PC: Intel Core i7 (9th gen), RTX 3060 Ti, 32 GB RAM.
  Training time and loss logs weren't kept. Say so; don't guess.
- Speed: forward pass about 110 ms on a 2-core CPU. The "~100 ms" in the
  README was an estimate. End-to-end time was never measured.

**Bugs found and fixed (2026-10-20)**
- The inference copy of the sun encoder didn't zero the hours without sun; the
  training code did. Fixing it raised accuracy on short days from 41–49% to
  58–65% exact.
- The browser put the sun vectors in fixed clock slots; training packs the
  daylight hours at the front. Fixed to match training.

**Symmetry probe** (`training/probe_symmetry.py`, 50 test scenes)
- Same scene, sun vectors in reverse order (the right answer doesn't change):
  exact accuracy drops from 59.8% to 29.0%.
- Mirrored scene with mirrored sun: 50.2% (east-west), 52.5% (north-south).

**About AI-written code.** Interviewers assume every design choice is yours.
If asked directly whether you used AI to write code, say yes, you used AI
coding tools, and you can explain every design decision. This file is how you
get there.

---

## Part 1: InstantSim

### A. The project and why

**1. Describe the project in one or two sentences.**

**Say:**
In urban design, checking how much sun a place gets means running a
simulation. That takes about a minute, so designers usually only check it at
the very end. I wanted the answer to show up while they're still designing. So
I trained a neural network to do it. You give it the building heights and the
date and place, and it gives you the sun-hours map in about a tenth of a
second. 90% of the pixels are within an hour of the real simulation.

**Follow-ups:**

**"What exactly is direct sun hours?"**
For each point on the ground or a roof, it's how many hours the sun can
actually reach it. Ladybug checks the sun at 9, 10, 11 and so on up to 17. At
each hour it draws a line from the point to the sun. If no building is in the
way, that hour counts. So the answer is between 0 and 9.

**"9 to 5 is only 8 hours. How can a point get 9?"**
Because Ladybug counts the sun at 9 time points, 9:00 to 17:00, and each one
counts as an hour. So really it's "how many of these 9 moments have sun". It's
an approximation of the real time in sun.

**"Why do designers care?"**
Sun on parks, playgrounds and courtyards makes a big difference to how
comfortable people are outside. Many cities also have rules about minimum sun
on open space. Right now it gets checked late because the simulation is slow.

**"Why 9 to 5 and not sunrise to sunset?"**
That's when people are mostly outside. It also keeps the answer between 0 and 9
for every day of the year, so the output is the same kind of thing in summer
and winter.

**"Is this research, or an application?"**
Honestly, the network itself is a known recipe, a pix2pix-style conditional
GAN. What I did was frame the problem: I feed the sun directions in as a
condition, so one model works for any place and date. I also built the data
pipeline and the whole tool. And when I evaluated it properly, I found where
the recipe breaks. That's what I'd want to work on next.

---

**2. Why train a model at all? Why not just make the ray tracing faster?**

**Say:**
That's a fair point. For sun hours, a GPU ray tracer would also be fast, and
it would be exact. I chose learning because the same approach carries over to
things that don't have a fast exact solver, like wind or surface temperature,
where a simulation can take hours. Sun hours was a good first target because
the right answer is cheap to compute and exact. So I could study the model
itself without worrying about bad labels.

**Follow-ups:**

**"So for sun hours alone, is the network worth it?"**
For accuracy, no. A GPU ray tracer is exact. Its value is as a test case with
perfect labels, and as a fast model you can take gradients through, so it can
sit inside an optimisation or a generator.

**"Why do gradients matter?"**
If you can take the gradient of sun hours with respect to the height map, an
optimiser can directly push the design toward more sun on open space. You
don't need trial and error.

**"What would change for wind?"**
A lot. Wind depends on the wind direction and speed instead of the sun. The
output is a velocity field, not a count. And the labels come from CFD, which
takes hours per case, so you'd have far fewer examples. Then generalisation
becomes the main problem. The conditioning part, FiLM with the wind direction,
would carry over.

---

**3. Where does the speed-up come from? How did you measure it?**

**Say:**
Ladybug draws a ray from every grid point to the sun, for every hour. So the
cost grows with the size of the site. The network just does one pass over the
image, and that cost is the same no matter how many buildings there are.
Ladybug takes somewhere between tens of seconds and a minute per scene. The
network's forward pass takes about 110 milliseconds on a 2-core CPU. So it's
a few hundred times faster. To be honest, I measured the forward pass, but not
the full round trip from Rhino to the screen.

**Follow-ups:**

**"How big is the network?"**
About 60 million parameters. The cost depends on the image size, not on how
complicated the scene is.

**"What takes the most time end to end?"**
Probably the screenshot in Rhino, the upload, and the forward pass. The
post-processing is small. I haven't measured it; that would be the first thing
I'd measure.

**"How would you make it faster?"**
Run it on the GPU, use half precision, or make the network smaller. 512
channels at the bottom is more than this task needs. Or train a smaller
student model from this one.

---

### B. Input 1: the height map

**4. Why use a height map as the input?**

**Say:**
The network works on images, so I needed the scene as a grid. A height map
from the top is the simplest way: each pixel stores how tall the building is
there. And that's exactly what casts shadows on the ground: where the buildings
are and how tall they are. I make it by painting the buildings in Rhino by
height, gray for height, and taking a screenshot from the top view.

How it's made (Grasshopper Python script):

```python
BUILDING_H_MIN = 0.0
BUILDING_H_MAX = 250.0
COLOR_AT_BOTTOM = 195        # gray at 0 m
COLOR_AT_TOP = 0             # gray at 250 m and above
COLOR_GROUND = 240
COLOR_BACKGROUND = 255

for v in joined_buildings.Vertices:          # colour every building vertex by its height
    z = v.Z
    if z < BUILDING_H_MIN: z = BUILDING_H_MIN
    if z > BUILDING_H_MAX: z = BUILDING_H_MAX   # clipped above 250 m
    t = (z - BUILDING_H_MIN) / (BUILDING_H_MAX - BUILDING_H_MIN)
    gray_val = int(COLOR_AT_BOTTOM + t * (COLOR_AT_TOP - COLOR_AT_BOTTOM))
    colors.append(System.Drawing.Color.FromArgb(gray_val, gray_val, gray_val))

# a huge plane with colour 255, 50 m below everything, is the background

# camera: Top view, Rendered mode, zoom to a square around the scene
max_dim = max(dx, dy)                         # longer side of the bounding box
half_side = (max_dim * ZOOM_FACTOR) / 2.0     # ZOOM_FACTOR = 1.2, a 20% margin
vp.ZoomBoundingBox(target_bbox)

capture = Rhino.Display.ViewCapture()
capture.Width = 512
capture.Height = 512
bitmap = capture.CaptureToBitmap(view)        # this PNG is sent to /predict
```

In the Grasshopper file the training scenes were coloured the same way:
vertex z → Remap 0..250 to 60..255 → Gradient white to black → Mesh Colours.
60 on the gradient is gray 195, 255 is black.

How the backend reads it (`inference.py`):

```python
def _build_valid_mask_from_input(self, input_bgr):
    # Background = 255 (outside the site) -> ignore
    # Ground = 240, buildings < 240 (darker = taller) -> keep
    gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    valid = (gray != 255)
    kernel = np.ones((3, 3), np.uint8)
    valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    return valid
```

**Follow-ups:**

**"How does height turn into gray?"**
Linearly. Ground is 240, background is 255. Buildings go from 195 at 0 m down
to 0 at 250 m, so darker means taller. I left a gap between 195 and 240 so the
lowest building never looks like ground. Anything taller than 250 m gets
clipped to 0. 250 m covers most sites.

**"Why start buildings at 195 and not 239?"**
To keep a clear gap from the ground. If a 2 m building were 238 and the ground
240, the network could easily mix them up. With the gap, every building is
clearly different from ground.

**"What does the screenshot actually store?"**
Not exactly the colours I set. Rendered mode adds a gamma curve, so the saved
values aren't linear in height. A 0 m building comes out around 226, 50 m
around 204, 125 m around 165, 250 m at 0. I checked this in the data: the
darkest values are exactly what you get by putting 0, 1, 2, 3… through that
curve. The network can still learn it, since it's monotonic, but tall
buildings get many gray levels and low buildings get few.

**"Is that a problem?"**
It's not ideal. The image depends on display settings, not just geometry.
Edges also get blended: there are values between ground and background that
no object was given. I checked that training images and the live tool's
images use the same encoding, so the model isn't confused at use time. But a
cleaner way is to cast a ray down for each pixel and save the height as a
number, in a 16-bit or float image.

**"How precise is the height?"**
250 m over 195 gray levels, so about 1.3 m per level. That's fine for building
shadows, but it can't tell a 1 m wall from flat ground.

**"What can't a height map show?"**
Anything with more than one height in the same spot: overhangs, arcades,
bridges, trees with space underneath. For sun on the ground that's usually
fine. For sun on facades it isn't.

**"Why three channels if it's grayscale?"**
Just to match the 3-channel pix2pix setup. It wastes two channels. Better would
be one channel for height and one for "where to analyse".

**"Is 255 for outside the site a good idea?"**
Not really. It mixes "don't analyse here" with "nothing is here". In a real
city, buildings next to the site cast shadows into it, and this encoding can't
show them. I'd use one channel for all the geometry, neighbours included, and
a separate mask for where to evaluate.

**"Why use the vertex z and not height above ground?"**
It was simpler. But it assumes the ground is at z = 0. On a sloped site, or a
model placed higher up, every building would be read wrong.

**"Does the network know the real height of a building?"**
Yes, because the mapping is fixed: 0 to 250 m, same for every scene. That
matters, because shadow length depends on real height.

**"Does the network know how many metres one pixel is?"**
No. And this is a real flaw. The camera always zooms so the site fills the
image. My training sites were 600 to 1300 m, so one pixel at 256 px is
anywhere from 2.8 m to 6.1 m. Heights are in metres, but shadows are measured
in pixels. So the same building with the same sun casts a shadow twice as
many pixels long on a small site as on a big one, and the model can't tell
which case it's in. It can only learn an average. That probably hurts most
for long shadows, so it may be part of why high-rise scenes are harder. I
haven't tested that yet.

**"Why did you make the site size random?"**
I didn't want the model to only know 1 km by 1 km sites. I wanted it to
generalise. The idea was right: varying the data helps avoid overfitting. But
the screenshot undid it. Every site gets zoomed to fill the image, so the
model never sees "a bigger site". It just sees the same-sized image at an
unknown scale. What I meant as variety turned into ambiguity.

**"How would you fix it?"**
Three options. Keep metres per pixel fixed, so a bigger site just covers more
of the image or gets split into tiles. Or give the scale to the model as an
extra input. Or store height in pixels instead of metres, height divided by
pixel size. Then shadow length in pixels is just height in pixels divided by
the tangent of the sun angle, whatever the site size.

---

**5. What if the image isn't 256 × 256?**

**Say:**
The backend resizes every input to 256 × 256 with nearest-neighbour, the same
as in training. After the prediction it scales the result up to 512 for
display.

```python
# inference.py
img_input_bgr_256 = cv2.resize(img_input_bgr, (256, 256),
                               interpolation=cv2.INTER_NEAREST)

# train_custom_solar.py, the same for the training input
transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
```

**Follow-ups:**

**"Why nearest and not bilinear?"**
Bilinear blends neighbouring pixels. At the edge of a 30 m roof next to the
ground, you'd get a pixel that looks like a 15 m building. Nearest keeps every
pixel a real height.

**"What if the image isn't square?"**
It gets stretched, which distorts footprints and shadows. The Grasshopper side
always makes a square image, but the backend doesn't check. It should pad to a
square instead.

**"Could you just run it at 512 or 1024?"**
Technically yes, it's fully convolutional. But the network never saw those
sizes in training, and shadow lengths in pixels change. So not without
retraining.

**"What about a bigger site?"**
Either each pixel covers more ground, so less detail, or you split it into
tiles. Tiles need to overlap, because a shadow can cross the border. The
overlap has to be at least as long as the longest shadow.

**"Why 256?"**
It's the standard pix2pix size and fits batch 16 in memory. The cost is
detail. At 2.8 to 6.1 m per pixel, a 20 m building is only 3 to 7 pixels wide,
and a 15 m street is 2 to 5. Courtyards and narrow streets, where sun matters
most for people, are just a few pixels. That's a big reason one-pixel errors at
shadow edges cost so much.

**"Can I use it for a small plaza?"**
Not as it is. It learned neighbourhood-size sites around 1 km. A 100 m site
would be zoomed to fill the image at 0.5 m per pixel, a scale it never saw.

---

**6. Does the site orientation matter?**

**Say:**
Yes. The sun directions are in world coordinates, so the image has to be
north-up, like in training. If you rotate the site in the image but not the
sun, the shadows come out wrong.

**Follow-ups:**

**"How did you check which way the image axes go?"**
With the symmetry test. If I flip the image top to bottom and flip the north
part of every sun vector, accuracy stays close to the original. That fits
image rows running along world y.

**"Could the model handle any rotation?"**
The physics allows it: rotate the scene and the sun together, and the answer
rotates the same way. You could teach that with augmentation, rotating both
together. Or you rotate everything so the sun always comes from the same side,
run the model, and rotate the result back.

---

### C. Input 2: the sun

**7. Why are the sun conditions 36 numbers?**

**Say:**
There are 9 hours, 9 to 17. For each hour I give the sun direction as 3
numbers, x, y and z. That's 27. Then 9 more numbers, one per slot, saying
whether there's actually a sun position there. In winter at high latitudes the
sun can set before 5, so some slots are empty.

```python
# SolarDataset: the log only has the hours with sun, in time order
vecs = entry.get('vectors', [])                 # list of [x, y, z], up to 9

raw_vecs = np.zeros((9, 3), dtype=np.float32)   # empty slots stay zero
mask = np.zeros((9,), dtype=np.float32)

num_to_fill = min(len(vecs), 9)
if num_to_fill > 0:
    raw_vecs[:num_to_fill] = np.array(vecs[:num_to_fill], dtype=np.float32)
    mask[:num_to_fill] = 1.0                    # sunny hours packed at the front

vec36 = np.concatenate([raw_vecs.flatten(), mask], axis=0)   # 27 + 9 = 36
```

A real entry (8 hours of sun; 17:00 is missing):

```
{"id": "0000", "vectors": [[-0.548, 0.7248, -0.4175], [-0.3194, 0.8071, -0.4965],
 ..., [0.9249, 0.3723, -0.0771]]}
```

**Follow-ups:**

**"What does one vector mean?"**
It's a unit vector pointing from the sun down to the ground. That's Ladybug's
convention, so z is negative. The first one here has z = −0.42, which means
the sun is about 25 degrees above the horizon.

**"Why give sun directions and not latitude and date?"**
Because shadows only depend on where the sun is, not why it's there. Two
different cities on different dates with the same sun path give the same
answer. Giving the vectors makes that obvious. With latitude and date, the
network would have to learn astronomy first.

**"Why the extra 9 flags, if empty slots are already zero?"**
A zero vector isn't a real direction, but the network doesn't know that. After
the first layer, a zero input becomes the bias, which isn't zero. The flags
tell the model which slots are real, and the encoder uses them to zero the
empty ones.

**"Is slot 0 always 9 o'clock?"**
No. The log only keeps hours with sun and packs them at the front. On a short
day slot 0 might be 10:00. So a slot means "the k-th sunny hour", not a clock
time. That's a weakness. The model can still tell morning from evening by the
direction, but it has to learn that.

**"Where do the vectors come from in the live tool?"**
The browser calculates the sun position with a library called SunCalc for the
chosen place and date, then converts it to Ladybug's convention. Training used
Ladybug's own sun path. They differ by tiny fractions of a degree, far below
what matters here.

**"Which time zone is 9 o'clock?"**
Local solar time. The browser uses longitude divided by 15 as the time zone,
no daylight saving. I checked the training data: on full days the sun is
highest in the 12:00 slot 90% of the time, so training used the same
convention. They match. But a user in Madrid, or in Toronto in summer, would
think of clock time, which can be an hour or two off.

**"Why are 44% of training days shorter than 9 hours?"**
Because the window is 3 hours before noon and 5 hours after. In winter at mid
and high latitudes, the sun sets before 5. That's why it's almost always the
evening that's missing, not the morning: 1,386 against 392 training scenes.

**"Why latitudes 35°S to 65°N?"**
It covers most cities. But it's lopsided. Nothing south of 35°S, so Melbourne,
Auckland and Wellington are outside. And it's uniform in latitude, not weighted
by where people live.

```javascript
// index.html: the 9 vectors sent to the backend
const dataTimeZoneOffset = Math.round(lon / 15);
for (let h = 0; h <= 24; h++) {
  if (h >= 9 && h <= 17) {
    const dUtcHour = h - dataTimeZoneOffset;
    const dPos = SunCalc.getPosition(new Date(baseTime + dUtcHour * 3600 * 1000), lat, lon);
    if (dPos.altitude > 0.0) {
      // ... convert to a unit vector, push x, y, z
    }
  }
}
// Same layout as training: sunny hours packed at the front,
// empty slots filled with zeros at the end
while (sunVectors.length < 27) sunVectors.push(0.0, 0.0, 0.0);
```

**"Why one sun position per hour, not more?"**
Because the label is hourly. Ladybug with a time step of 1 checks once per
hour. Finer steps would give a smoother answer, but then I'd need more slots,
or an encoder that doesn't care how many vectors there are (question 14).

---

**8. How do 36 numbers get into an image network?**

**Say:**
A small encoder turns them into 128 numbers. Each sun direction goes through
the same small network. Hours without sun get zeroed. Then I put the 9
results side by side, in order, add the flags and the fraction of sunny hours,
and one layer squeezes it all into 128 numbers. Every FiLM layer in the main
network reads those 128 numbers.

```python
class SunConditionEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.per_vec = nn.Sequential(               # shared by all 9 hours
            nn.Linear(3, emb_dim), nn.LeakyReLU(0.2, True),
            nn.Linear(emb_dim, emb_dim), nn.LeakyReLU(0.2, True),
        )
        fuse_in = 9 * emb_dim + 9 + 1               # 1152 + 9 flags + 1 day length
        self.fuse = nn.Sequential(nn.Linear(fuse_in, emb_dim), nn.LeakyReLU(0.2, True))

    def forward(self, vec):                          # vec: (B, 36)
        v = vec[:, :27].view(-1, 9, 3)
        m = (vec[:, 27:].view(-1, 9, 1) > 0.5).float()
        feat = self.per_vec(v) * m                   # (B, 9, 128), empty hours = 0
        feat_flat = feat.reshape(feat.size(0), -1)   # keep the order: (B, 1152)
        mask_flat = m.view(m.size(0), 9)
        daylen = mask_flat.sum(dim=1, keepdim=True).clamp(min=1.0) / 9.0
        return self.fuse(torch.cat([feat_flat, mask_flat, daylen], dim=1))
```

**Follow-ups:**

**"Why use the same small network for all 9 hours?"**
Turning a direction into features is the same job every hour. Sharing it means
9 times fewer weights, and every training example teaches it 9 times. Which
hour it is comes from the position in the line-up.

**"Why the day-length number, when the flags already say it?"**
It's redundant, just a shortcut. It's the number of flags divided by 9, so the
layer doesn't have to count. Not essential.

**"Why 128?"**
A common size. 36 inputs don't need more. I didn't tune it.

**"Is one set of 128 numbers enough for the whole image?"**
For the sun, yes, because the sun is the same for every pixel. If the
condition changed across the image, like different materials in different
places, you'd need a spatial method, like SPADE.

---

**9. Tell me about a bug you found.**

**Say:**
When I evaluated the model properly, short winter days were clearly worse than
full days. That pointed to the code that only matters on short days. I
compared the training code and the inference code line by line and found the
inference version was missing one line. In training, the hours without sun
were zeroed. In inference they weren't, so empty hours fed in values the model
had never seen. After the fix, accuracy on short days went from about 45% to
about 60% exact.

```python
# backend/app/models/networks.py, after the fix
feat = self.per_vec(v)
# Zero the hours without sun, as in training
# (per_vec(0) is not zero because of the biases)
feat = feat * m01.unsqueeze(-1)
```

**Follow-ups:**

**"Why didn't it break full days?"**
On a full day all 9 flags are 1, so multiplying by them changes nothing.

**"How would you stop this happening again?"**
Have one model definition that both training and inference import, instead of
two copies. And a test that runs both on the same input and checks they give
the same output.

**"Was there anything else like that?"**
Yes. The browser put the sun in fixed clock slots, 9:00 always in slot 0, with
a zero if the sun was down. Training packs the sunny hours at the front. I
fixed the browser to match. It only mattered on days when the sun is down at 9.

---

### D. Output

**10. What does the model output? Why a colour image?**

**Say:**
It outputs a colour image, 256 × 256, in Ladybug's legend colours. 10 colours
for 0 to 9 hours. Honestly, I did it that way because Ladybug's coloured image
was the easiest label to get, and it fits pix2pix directly. Afterwards I turn
the colours back into hours by matching each pixel to the nearest legend
colour.

```python
self.last = nn.Sequential(
    nn.ReLU(True),
    nn.ConvTranspose2d(128, output_nc, 4, 2, 1),   # 128 -> 256 pixels, 3 channels
    nn.Tanh()                                      # [-1, 1], same range as the label
)
```

**Follow-ups:**

**"What's wrong with predicting colours?"**
The loss is computed on colours, not hours. The legend goes blue, purple, red,
orange, yellow, so the distance between two colours doesn't match the
difference in hours. And it uses 3 channels for what is really one number.
Errors land between colours and have to be snapped back.

**"What would you predict instead?"**
Three options. One number per pixel, the hours, with an L1 loss in hours.
Simple. Or 10 classes per pixel with cross-entropy, which gives a probability
for each, so a built-in confidence, but it doesn't know 3 is closer to 4 than
to 9. Or ordinal: predict "more than 0 hours", "more than 1 hour", and so on.
That respects the order and suits a count best.

**"Anything better than that?"**
Use the physics. Sun hours is just "lit at 9" plus "lit at 10" and so on. So
predict 9 shadow masks, one per hour, and add them up. Each mask is an easier
problem, one sun direction. The total can't go above the number of sunny
hours. The order of the hours stops mattering. And you can check each hour's
shadow on its own.

**"Why Tanh at the end?"**
The labels are scaled to −1 to 1, and Tanh outputs exactly that range.

**"Can it predict more hours than the sun is up?"**
In principle, yes. In practice, only 0.7% of pixels on short days go over. So
it has mostly learned the limit. The shadow-mask version would guarantee it.

---

**11. Why are labels resized with bilinear but height maps with nearest?**

**Say:**
That's in the training code: nearest for the height map to keep sharp edges,
bilinear for the label. Looking back, bilinear on the label was a mistake.
Averaging two legend colours gives a colour that isn't in the legend, so I was
training the network to output in-between colours at the edges.

```python
self.transform_A = transforms.Compose([...,
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST), ...])
self.transform_B = transforms.Compose([...,
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR), ...])
```

**Follow-ups:**

**"Does it matter much?"**
Probably a little, at the edges, which is where most errors are. The original
Ladybug images already have blended colours too: only about 82% of pixels are
exact legend colours. In evaluation I resize labels with nearest and snap every
pixel to a legend colour, so the numbers are in clean classes.

**"What would you do?"**
Turn the label into hour numbers first, then resize with nearest.

---

### E. Network

**12. Why a U-Net? Walk me through it.**

**Say:**
The output is a map the same size as the input, lined up pixel by pixel. A
U-Net shrinks the image in 8 steps, from 256 down to 1 pixel, so it can see
the whole scene. Then it grows it back in 8 steps. At each size there's a skip
connection that passes the details straight across, so the building edges stay
sharp.

```python
def forward(self, x, vec):
    emb = self.cond_enc(vec)                 # sun -> 128 numbers

    e1 = self.enc1(x)                        # 256 -> 128, no norm, no FiLM
    e2 = self.enc2(e1, emb)                  # each step: LeakyReLU, conv (halves size), FiLM
    e3 = self.enc3(e2, emb)                  # 64
    e4 = self.enc4(e3, emb)                  # 32
    e5 = self.enc5(e4, emb)                  # 16
    e6 = self.enc6(e5, emb)                  # 8
    e7 = self.enc7(e6, emb)                  # 4 ... 2
    e8 = self.enc8(self.enc8_act(e7))        # 1×1, no norm, no FiLM

    b, c, h, w = e8.size()                   # the sun also enters here as 512 extra channels
    v = self.bottleneck_fc(emb).view(b, 512, 1, 1).expand(b, 512, h, w)

    d1 = self.dec1(torch.cat([e8, v], 1), emb)
    d1 = torch.cat([d1, e7], 1)              # skip connection
    d2 = self.dec2(d1, emb); d2 = torch.cat([d2, e6], 1)
    d3 = self.dec3(d2, emb); d3 = torch.cat([d3, e5], 1)
    d4 = self.dec4(d3, emb); d4 = torch.cat([d4, e4], 1)
    d5 = self.dec5(d4, emb); d5 = torch.cat([d5, e3], 1)
    d6 = self.dec6(d5, emb); d6 = torch.cat([d6, e2], 1)
    d7 = self.dec7(d6, emb); d7 = torch.cat([d7, e1], 1)
    return self.last(d7)                     # -> 3 × 256 × 256
```

**Follow-ups:**

**"Why does a shadow problem need to see the whole scene?"**
Because shadows can be long. Shadow length is height divided by the tangent of
the sun angle. At 10 degrees, a building casts a shadow almost 6 times its
height. The training data has sun as low as a couple of degrees. So a pixel
can be shaded by a building far away. The 1 × 1 bottleneck sees the whole
image.

**"Then why the skip connections?"**
The bottleneck knows roughly "there's a tall building to the south-west". The
skip connections know exactly which pixels are its edge. The decoder needs
both.

**"Why 4 × 4 convolutions with stride 2, not pooling?"**
That's the pix2pix design. The network learns how to shrink instead of using a
fixed max pool. A known side effect is a checkerboard pattern on the way back
up.

**"Why does enc8 have no normalisation?"**
It's 1 × 1. Instance norm over one pixel subtracts the value from itself and
gives zero, which would wipe out the bottleneck. enc1 has none either, like in
pix2pix, so the first skip keeps the raw input values.

**"What does dropout do here?"**
Dropout 0.5 in the first three decoder layers, during training only, like
pix2pix. In use, `model.eval()` turns it off, so the output is always the same
for the same input. If I turned it back on and ran the model 20 times, the
spread would give a cheap confidence estimate. That's called Monte Carlo
dropout. I haven't tried it.

**"How big is it?"**
60.1 million parameters, a bit more than the original pix2pix U-Net, about 54
million, because of the FiLM layers and the sun encoder. That's large for 4,000
training images, which is one reason training accuracy is higher than test.

**"What else could you use?"**
A smaller U-Net. Attention in the middle for long-range effects. A vision
transformer. Or build in the geometry: a shadow is just the height map pushed
along the sun direction, so you could do that projection directly in the
first layers.

---

**13. What is FiLM, and why use it?**

**Say:**
FiLM lets the sun decide how each feature channel is scaled and shifted. In
each layer, I normalise the features, then multiply by a gamma and add a beta,
and gamma and beta come from the sun's 128 numbers. I use it in 13 layers, so
the sun affects both the big picture and the fine details.

```python
class FiLMIN(nn.Module):
    def __init__(self, num_features, emb_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=1e-5)
        self.to_gamma_beta = nn.Linear(emb_dim, num_features * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)   # gamma = 1, beta = 0 at the start
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, emb):
        b, c, _, _ = x.shape
        x = self.norm(x)                              # per image, per channel
        gb = self.to_gamma_beta(emb)
        gamma, beta = gb[:, :c], gb[:, c:]
        gamma = gamma.view(b, c, 1, 1) + 1.0
        beta = beta.view(b, c, 1, 1)
        return gamma * x + beta
```

**Follow-ups:**

**"Why start at zero and add 1?"**
So at the start gamma is 1 and beta is 0, and FiLM does nothing. The network
starts as a plain U-Net, and the sun's effect grows as training goes. If gamma
started random, some channels could be almost switched off or blown up, and GAN
training is already unstable.

**"Why not just add the sun to the input image as extra channels?"**
That works too, but then the sun information has to survive through every
layer. FiLM puts it in at every level. Also, 36 constant channels of 256 × 256
is a lot of wasted input.

**"Why instance norm under FiLM?"**
Instance norm removes each image's own channel statistics. Then FiLM sets them
based on the sun. It's the same idea as style transfer, where the style decides
the statistics. Batch norm would mix statistics from images with different
suns in the same batch.

**"Does instance norm throw anything useful away?"**
It can. The average of a feature over the image, like "how built-up is this
scene", gets removed in every normalised layer, and only comes back through
beta, which depends on the sun, not the scene. The first skip connection and
the bottleneck keep some of it. Worth testing.

**"How do you know the model actually uses the sun?"**
From the symmetry test. Same height map, same sun vectors, just in reverse
order, and the output changes a lot. So the sun clearly drives the output. A
cleaner test would be: same scene, morning-only sun versus afternoon-only sun,
and check the shadows switch sides.

**"Alternatives to FiLM?"**
Cross-attention: each pixel looks at the 9 sun vectors and picks the hours that
matter to it. That would also stop the order from mattering. Or SPADE, if the
condition changed across the image.

---

**14. Why keep the 9 sun vectors in order instead of averaging them?**

**Say:**
If you average them, the whole day collapses into one direction. You lose
"sunny in the morning, shaded in the afternoon", and very different days can
average to the same thing. Keeping them in order keeps everything. But my
symmetry test showed a cost. The right answer doesn't depend on the order, but
the model does. If I just reverse the order, accuracy halves. So the right
design is something that doesn't care about order, but still keeps the
information.

Results of `training/probe_symmetry.py` (50 test scenes):

| What I changed | Right answer | Exact | Within 1 h |
|---|---|---|---|
| nothing | same | 59.8% | 91.1% |
| sun vectors in reverse order | same | 29.0% | 72.7% |
| mirror east-west (flip image left-right, flip x of the sun, reverse order) | mirrored | 50.2% | 84.7% |
| mirror north-south (flip image top-bottom, flip y of the sun) | mirrored | 52.5% | 86.4% |

```python
def transform(A, B, vectors, variant):
    v = [list(x) for x in vectors]
    if variant == "reversed_order":
        v = v[::-1]
    elif variant == "mirror_east_west":
        A, B = A[:, ::-1], B[:, ::-1]
        v = [[-x, y, z] for x, y, z in v][::-1]
    elif variant == "mirror_north_south":
        A, B = A[::-1], B[::-1]
        v = [[x, -y, z] for x, y, z in v]
    return np.ascontiguousarray(A), np.ascontiguousarray(B), v
```

**Follow-ups:**

**"What would an order-free design look like?"**
Run each sun vector through a small network, add up the results, then pass the
sum through another small network. That's called DeepSets. It's different from
averaging the raw vectors: each direction first becomes many features, so
nothing is lost. And the result is the same in any order.

**"Why is order-free the right physics?"**
Because sun hours is literally a sum over the hours: lit at 9, plus lit at 10,
and so on. A sum doesn't care about order. The shadow-mask idea from question
10 is the most direct version of this.

**"Why did reversing hurt so much?"**
In training, slot 0 always held the earliest sunny hour. So the network learned
weights that are tied to each slot. Reversed input is something it has never
seen. It learned a shortcut that holds in the data but not in the physics.

**"Why are the mirrored scenes worse, when they're valid?"**
Partly because the model hasn't seen those sun paths. Training covers 35°S to
65°N. If you mirror a scene at 50°N north-south, it looks like a site at 50°S,
which isn't in the training data. So it tells me how much the model learned
geometry versus the sun paths it happened to see.

**"How would you fix it cheaply?"**
Augmentation that respects the physics. Rotate or mirror the scene together
with the sun, and shuffle the order. It costs nothing, because the right
answer transforms exactly the same way, no new simulation needed.

---

### F. Loss and discriminator

**15. What's your loss?**

**Say:**
Two parts. An L1 loss, the average pixel difference from the real answer,
times 50. And a GAN loss, which is a second network, the discriminator, trying
to tell my output from a real Ladybug result. L1 keeps the values right. The
GAN part pushes the edges to be sharp.

```python
criterionGAN = nn.MSELoss()      # least-squares GAN; pix2pix uses cross-entropy
criterionL1 = nn.L1Loss()
L1_LAMBDA = 50.0                 # pix2pix uses 100

fake_B = netG(real_A, vec)
pred_fake_for_g = netD(real_A, fake_B, vec)
loss_G_GAN = criterionGAN(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
loss_G_L1 = criterionL1(fake_B, real_B) * L1_LAMBDA
loss_G = loss_G_GAN + loss_G_L1
```

**Follow-ups:**

**"Why L1 and not L2?"**
When the model isn't sure between two values, L2 picks the average and L1
picks the middle one. The average of "shadow" and "no shadow" is a grey blur.
L1 picks one of them, so it's less blurry. It also isn't dominated by a few big
errors.

**"Why a GAN at all, if L1 works?"**
L1 alone still blurs shadow edges where the model is unsure. The discriminator
rewards outputs that look like real Ladybug maps, and real maps have sharp
edges.

**"Is a GAN the right tool for a physics model?"**
Honestly, I'm not sure it is. A GAN rewards looking real, not being right. It
can draw a sharp edge in the wrong place, and that looks better than a blurry
edge in the right place. For a physics surrogate, sharpness only helps if it's
accurate. Now I'd compare against L1 alone, and against a classification loss,
which avoids the blur without a second network.

**"Why least-squares instead of the normal GAN loss?"**
With the normal loss, once the discriminator is confident, the generator's
gradient almost disappears. The least-squares version keeps pushing in
proportion to how far off you are, so training is usually more stable. I
didn't test it against the original.

**"Why 50 instead of 100?"**
A lower L1 weight gives the GAN part more say, aiming for sharper edges. It was
a judgement call, not tuned. The obvious experiment is to try GAN only, 10,
50, 100, and L1 only.

**"How big were the two terms during training?"**
The script prints both, but I didn't keep the logs. Next time I'd log both
every epoch.

---

**16. What does the discriminator look like?**

**Say:**
It's a conditional PatchGAN. It looks at the height map, a sun-hours map, real
or generated, and the sun. Instead of one real-or-fake score, it gives a 30 ×
30 grid of scores, each judging a 70 × 70 pixel patch.

```python
class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_nc=6, vec_dim=36, emb_dim=128, cond_ch=16):
        super().__init__()
        self.cond_enc = SunConditionEncoder(emb_dim=emb_dim)   # its own sun encoder
        self.cond_to_map = nn.Sequential(nn.Linear(emb_dim, cond_ch), nn.LeakyReLU(0.2, True))
        in_c = input_nc + cond_ch                               # 3 + 3 + 16 = 22
        self.model = nn.Sequential(
            nn.Conv2d(in_c, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),                  # 128
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),  # 64
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True), # 32
            nn.Conv2d(256, 512, 4, 1, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True), # 31
            nn.Conv2d(512, 1, 4, 1, 1)                                                     # 30
        )

    def forward(self, x, y, vec):
        emb = self.cond_enc(vec)
        c = self.cond_to_map(emb).view(emb.size(0), -1, 1, 1)
        c = c.expand(-1, c.size(1), x.size(2), x.size(3))   # same 16 values at every pixel
        return self.model(torch.cat([x, y, c], dim=1))
```

**Follow-ups:**

**"Where does 70 × 70 come from?"**
Work backwards from one output pixel. The last two layers have kernel 4 and
stride 1, so 4, then 7. Each stride-2 layer doubles it and adds a bit: 16, 34,
70.

**"Why patches instead of one score?"**
L1 already takes care of the big picture. The discriminator only needs to judge
local things, like whether edges are sharp. A patch discriminator is smaller
and works on any image size.

**"But shadows are long. Can a 70-pixel patch judge them?"**
No. It can judge whether an edge is sharp, not whether a 150-pixel shadow is
the right length for this sun. That's another reason the GAN part is about
looks, not physics.

**"Why give it the height map and the sun too?"**
Without them, it could only ask "does this look like a sun-hours map?", and
the generator could output any realistic map. With them, it asks "is this the
right map for this scene and this sun?".

**"Why does it have its own sun encoder?"**
So the generator and discriminator don't share weights that they're training
in opposite directions.

---

### G. Training

**17. What were your training settings?**

```python
BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 200
SAVE_EPOCH_FREQ = 5
L1_LAMBDA = 50.0

optG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
optD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
```

**Say:**
Adam with learning rate 1e-4 for both networks, beta1 of 0.5, batch size 16,
200 epochs, and the learning rate stays the same the whole time. 4,000 images
in batches of 16 is 250 steps an epoch, so the released model, epoch 195, saw
about 49,000 updates. I trained it on my own PC with an RTX 3060 Ti.

**Follow-ups:**

**"Why beta1 of 0.5?"**
Beta1 is how much Adam remembers past gradients. In a GAN, the other network
keeps changing, so old gradients go stale fast. Less memory reacts faster and
wobbles less. It's the standard for DCGAN and pix2pix.

**"How is that different from pix2pix?"**
pix2pix uses learning rate 2e-4, batch size 1, and after 100 epochs it lowers
the learning rate step by step to zero. I used a lower rate, a bigger batch,
and no decay. The decay usually helps at the end, so not having it is a miss.

**"Why batch 16 with instance norm?"**
Instance norm doesn't depend on the batch, so batch size only changes how
noisy the gradients are and how fast it trains. 16 fits in memory at 256 px.

**"How long did training take?"**
I didn't record it. It was on one desktop: an i7, an RTX 3060 Ti, 32 GB RAM.

**"Why 200 epochs? How do you know it wasn't over- or under-trained?"**
200 is the pix2pix default. I didn't have a validation set, so I can't show
which epoch was best. Training accuracy is 70% exact versus 59% on test, so
there is some overfitting. With a validation split, I'd pick the checkpoint by
within-1-hour accuracy on it.

---

**18. Walk me through one training step.**

```python
for i, batch in enumerate(dataloader):
    real_A = batch['A']; real_B = batch['B']; vec = batch['vec']

    # 1) discriminator
    optD.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_B = netG(real_A, vec)                  # no graph for G here
    pred_real = netD(real_A, real_B, vec)           # should say 1
    pred_fake = netD(real_A, fake_B.detach(), vec)  # should say 0
    loss_D = (criterionGAN(pred_real, torch.ones_like(pred_real)) +
              criterionGAN(pred_fake, torch.zeros_like(pred_fake))) * 0.5
    loss_D.backward()
    optD.step()

    # 2) generator
    optG.zero_grad(set_to_none=True)
    fake_B = netG(real_A, vec)                      # again, now with a graph
    pred_fake_for_g = netD(real_A, fake_B, vec)     # try to make D say 1
    loss_G = criterionGAN(pred_fake_for_g, torch.ones_like(pred_fake_for_g)) \
             + criterionL1(fake_B, real_B) * L1_LAMBDA
    loss_G.backward()
    optG.step()
```

**Say:**
They take turns. First the discriminator: the generator makes a fake, the
discriminator sees a real pair and a fake pair, and learns to say 1 for real
and 0 for fake. Then the generator: it makes a fake again, and is trained to
make the discriminator say 1, plus stay close to the real answer through L1.

**Follow-ups:**

**"Why run the generator twice?"**
The first run is only to give the discriminator something to look at, so it
skips tracking gradients to save memory. The second run tracks them, because
the generator needs them to update. The more common way is one run with
`detach()`, which saves a forward pass.

**"Why multiply the discriminator loss by 0.5?"**
pix2pix does that to slow the discriminator down a bit compared to the
generator.

**"Does the generator step change the discriminator?"**
The backward pass computes gradients for the discriminator too, but only the
generator's optimiser takes a step. The discriminator's gradients are cleared
at the start of the next step.

---

**19. Was GAN training unstable? How did you keep track of it?**

**Say:**
I watched the printed losses, and every 5 epochs the script saved the input,
the prediction and the real answer for one fixed training image into a web
page, so I could see it improve. I don't remember specific problems, and I
didn't keep the logs. Looking back, that monitoring was weak.

```python
if epoch % SAVE_EPOCH_FREQ == 0 or epoch == EPOCHS - 1:
    netG.eval()
    with torch.no_grad():
        fixed_fake = netG(fixed_A, fixed_vec)
    save_visuals(epoch, fixed_id, fixed_A, fixed_fake, fixed_B, IMG_DIR)
    update_html(WEB_DIR, IMG_DIR)
    netG.train()
```

**Follow-ups:**

**"What's wrong with watching one training image?"**
It's a training image, so memorising looks like progress. And one image isn't
a measurement. GAN losses don't tell you much about quality either: both can
stay flat while the output gets better or worse.

**"What should you watch instead?"**
Within-1-hour accuracy and mean error on a held-out validation set every few
epochs. Plus a fixed set of validation images: short days, low sun, high-rise.

**"What usually goes wrong with GANs?"**
The discriminator wins too early and the generator stops learning. Or the two
keep swinging back and forth. Or mode collapse, where the generator gives the
same output for different inputs. That's less of a problem in pix2pix, because
L1 ties every output to its own answer.

---

### H. Data

**20. How did you make the dataset?**

**Say:**
I wrote a Grasshopper script. It starts from a 1 km square site, stretches it
randomly, splits it into plots, and extrudes each plot into a building, with
height, density and footprint shape sampled by category. Then it picks a random
place between 35°S and 65°N and a random date, and runs Ladybug for 9 to 5. It
saves the height map and Ladybug's image side by side, plus the sun vectors in
a log. 4,000 scenes for training and 200 for testing.

```python
# each training file is one 1024 × 512 PNG: height map left, label right
w_half = img.shape[1] // 2
real_A, real_B = img[:, :w_half, :], img[:, w_half:, :]
vec = self.vector_map.get(img_id, torch.zeros(VEC_DIM, dtype=torch.float32))
```

What's in the data (measured):

| | Training (4,000) | Test (200) |
|---|---|---|
| Days with all 9 hours of sun | 56% | 84% |
| Days with 3–6 hours of sun | 13% | 4.5% |
| Typical highest sun of the day (median) | 37.6° | 54.6° |
| Site share of the image | | 19–44% (median 30%) |
| Buildings' share of the site | | 2.5–30% (median 13%) |
| Height mix | mixed | 51 low, 49 mid, 100 high-rise |

**Follow-ups:**

**"Is the test set like the training set?"**
Not quite. The test set has many more long summer-like days, 84% versus 56%,
and the sun is higher. Low sun means long shadows, the hardest case. So the
test set is probably a bit easier, and the numbers may be optimistic for
winter.

**"Are test and training scenes independent?"**
The test scenes have new building layouts, but they reuse the site outlines
and sun conditions of training scenes. So the numbers show the model handles
buildings it hasn't seen, not new sites, places or dates. A cleaner test would
hold out whole locations and dates, or a whole building category.

**"Does it work on real cities?"**
I can't say yet. All my scenes come from my own generator. The biggest missing
test is real city blocks, for example from OpenStreetMap footprints with
heights, labelled with Ladybug.

**"Why 4,000?"**
It wasn't chosen carefully. It's what was practical to generate at up to a
minute per scene. A learning curve, training on 500, 1,000, 2,000 and 4,000,
would show whether more data still helps.

**"What settings did Ladybug use?"**
A 5 m grid, points 0.2 m above the surface, one sun position per hour.

**"What does a 5 m grid mean for the label?"**
The right answer itself only has one value every 5 m, and one model pixel is
2.8 to 6.1 m. So the label is about as coarse as the model, and its shadow
edges are only right to about one cell. Some of the one-hour errors at edges
are really the model disagreeing with a coarse label. The model can't be
judged more precisely than its labels.

**"Why not a 1 m grid?"**
With sites up to 1.3 km, a 1 m grid is about 1.6 million points per scene, 25
times the work of 5 m, for thousands of scenes. 5 m kept it affordable. The
price is a coarser label.

**"Why 0.2 m above the surface, when Eddy3D uses 0.918 m?"**
They answer different questions. Sun-hour rules look at sun on the ground and
open space, so the points sit just above the surface; the 0.2 m only stops the
ray hitting the surface it starts on. Eddy3D's sky exposure is about how a
standing person feels, so it starts at the middle of a person's height.

**"Are roofs part of the label?"**
Yes. Roofs are 13% of the site pixels, mostly 8 or 9 hours. Surprisingly the
model does worse on roofs than on the ground, mean error 0.92 versus 0.50
hours. My guess is roof edges: on a small roof, a one-pixel mismatch between
height map and label is a big part of the roof. I haven't tested that.

**"Why no augmentation?"**
I didn't do it. The right kind here moves the sun with the scene: rotate the
image and rotate the sun vectors the same way, or mirror both. Colour changes
or random crops would be wrong, because cropping changes what can cast a
shadow into the frame.

---

### I. Evaluation

**21. How did you measure accuracy?**

**Say:**
I ran the model on the 200 test scenes. Both the prediction and Ladybug's
answer are colour images, so I turn each pixel back into hours by finding the
closest legend colour. Then I compare pixel by pixel, only inside the site. I
report three things: exactly right, within one hour, and the average error in
hours. And I compare against a baseline that ignores buildings and gives every
pixel full sun.

```python
LEGEND_HEX = ["0000ff", "3a00c4", "770088", "b2004c", "e80014",
              "ff2b00", "ff6500", "ffa100", "ffdc00", "ffff00"]   # index = hours

def to_hours(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 1, 3).astype(np.float32)
    d = ((lab - LEGEND_LAB[None]) ** 2).sum(-1)       # distance to each legend colour
    return d.argmin(1).reshape(rgb.shape[:2])

site = ~np.all(A_s == 255, axis=2)                    # inside the site
t = to_hours(B_s)[site]
p = to_hours(pred_rgb)[site]
err = np.abs(p - t)
maxh = min(len(e["vectors"]), 9)
base = np.full_like(t, maxh)                          # baseline: full sun everywhere
```

| Over 3.97 million site pixels | Model | No-shadow baseline |
|---|---|---|
| Exactly right | 59.3% | 39.8% |
| Within 1 hour | 90.8% | 53.6% |
| Mean error | 0.56 h | 2.18 h |

**Follow-ups:**

**"Why match colours in Lab and not RGB?"**
In RGB, the distance between two colours doesn't match how different they
look. Lab is designed so it roughly does. The model's colours never exactly
hit the legend, so I need the closest match to be reliable.

**"Why that baseline?"**
It's what you get if you ignore the buildings, so the gap shows how much the
model learned about shadows. A stronger baseline would be more convincing,
like a simple geometric shadow projection, or a plain U-Net with L1 only.

**"Why is exact accuracy only 59%?"**
Sun hours change in one-hour steps across a shadow edge, so if an edge is off
by one pixel, that pixel is off by an hour. 76% of site pixels are within 2
pixels of an edge. There, it's 49% exact. Away from edges, it's 91%. For
design, within one hour and the mean error matter more.

**"Is that per pixel or per image?"**
Per pixel, over the whole test set. Averaged per image it's 58.5% exact, and
the worst image is 38%. So scenes do vary.

**"Is half an hour of error good enough for design?"**
Depends on the decision. If the question is "is this courtyard above 2 hours",
what matters is the error near 2 hours. A better metric would be: how much of
the area lands on the right side of the threshold.

**"What about statistical uncertainty?"**
With 200 scenes, I'd resample scenes, a bootstrap, to get a confidence
interval. The variation between scenes matters more than the pixel count. I
haven't done it.

---

**22. Where does the model go wrong?**

**Say:**
At shadow edges, in high-rise scenes, on roofs, and in shaded areas, where it
tends to predict too much sun.

| Where | Exact | Within 1 h | Mean error |
|---|---|---|---|
| Near shadow edges (76% of pixels) | 49.2% | 88.2% | 0.70 h |
| Away from edges (24%) | 90.8% | 99.2% | 0.10 h |
| Low- and mid-rise scenes (100) | 71.4% | 94.1% | 0.40 h |
| High-rise scenes (100) | 46.8% | 87.5% | 0.73 h |
| Ground (86%) | 60.4% | 92.3% | 0.50 h |
| Roofs (13%) | 52.9% | 81.2% | 0.92 h |
| Training scenes (49) | 70.3% | 95.4% | 0.38 h |

```python
# evaluate_model.py: pixels where the true answer changes within 2 px
T8 = T.astype(np.uint8)
band = (cv2.dilate(T8, kernel) != cv2.erode(T8, kernel)) & site   # kernel 5×5
for name, msk in (("near_class_boundary", band), ("uniform_region", site & ~band),
                  ("ground", ground), ("roof", roof)):
    er = np.abs(P[msk] - T[msk])
    ...
sc = scenes["high_rise" if (A_s[..., 0] < 128).any() else "low_rise"]
```

**Follow-ups:**

**"Why are high-rise scenes harder?"**
Longer shadows, shadows from several buildings overlapping, and more pixels in
partial shade, which are the hardest values. Also, towers above 250 m all look
like 250 m. And the unknown scale probably hurts long shadows the most.

**"Does it lean one way?"**
Yes. Where the truth is 0 to 3 hours, it predicts too high much more often
than too low. So shaded areas come out too sunny. From 4 to 8 hours, errors go
both ways about equally. 9 hours is the easiest, 82% right, because that's no
shadow at all and a third of all pixels.

**"Why would shade come out too sunny?"**
Imbalance. A third of pixels are 9 hours, only 4% are 0 hours, so the model
leans sunny. A weighted loss, or the per-hour shadow masks, would help.

**"Is the training-test gap overfitting?"**
Partly. 11 points in exact accuracy, under 5 within one hour. But the test set
is also distributed differently, which muddies the comparison.

**"What does the symmetry test add?"**
It checks something accuracy can't: whether the model behaves like the physics
on inputs that should be just as easy. The order problem is a design flaw, not
noise.

---

### J. The running tool

**23. How does the whole system work?**

**Say:**
When the design changes in Rhino, Grasshopper paints the buildings by height,
takes a screenshot from the top, and sends it to a Python backend. A second
script sends the 3D mesh so the browser can show it. In the browser you pick a
place and a date; it works out the sun directions and sends them over a
WebSocket. The backend runs the model, converts the colours to the display
palette, works out some statistics, and pushes the image and the numbers back
to the browser, which drapes the result over the 3D model.

```python
# inference.py, core of predict()
img_input_bgr_256 = cv2.resize(img_input_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
vec_batch = self.sun_vectors.unsqueeze(0)                     # (1, 36) from the browser

with torch.no_grad():
    output_tensor = self.model(input_tensor, vec_batch)       # model.eval()

valid_mask = self._build_valid_mask_from_input(img_input_bgr_256)
output_bgr_remapped = self._remap_colors(output_bgr, valid_mask)          # legend -> display
stats = self._calculate_stats_discrete(output_bgr_remapped, valid_mask)   # avg, %<2h, %>6h
```

**Follow-ups:**

**"How does the mesh get to the browser?"**
As JSON: a list of vertex coordinates and a list of triangles. Quads get split
into two triangles.

```python
for f in rhino_mesh.Faces:
    if f.IsQuad:                     # split a quad into two triangles
        faces.extend([f.A, f.B, f.C])
        faces.extend([f.A, f.C, f.D])
    else:
        faces.extend([f.A, f.B, f.C])
payload = {"vertices": verts, "faces": faces}
client.UploadString(url, "POST", json.dumps(payload))   # url = .../sync_model
```

That's fine for a design model. For a whole city I'd send a binary format like
glTF, and only when it changes.

**"Why load the model once at startup?"**
Loading 60 million parameters takes seconds. Doing it on every request would
kill the speed.

**"Why WebSocket and not polling?"**
Results get pushed to every open browser as soon as they're ready, in both
directions, without asking again and again.

**"What if two updates arrive while the model is busy?"**
They're handled in order, and each one runs the model. There's no "skip old
requests". For live editing, keeping only the newest would be better.

**"Why shrink the mask by one pixel?"**
Pixels right at the site border are a mix of site and background, so their
predictions aren't reliable. Shrinking by one pixel leaves them out of the
statistics.

**"Why the blur and the saturation boost?"**
The blur cuts single-pixel noise in the counts. The saturation boost makes the
display easier to read. But both happen before the dashboard statistics, so
those numbers can differ slightly from the evaluation script. It would be
cleaner to compute statistics from the hour values first.

**"How do you stop people trusting a wrong prediction?"**
Right now, nothing. On an unfamiliar design it still gives a believable map.
Options: check whether the input looks like the training data, give a
per-pixel confidence with Monte Carlo dropout or several models, and add a
button to run the real simulation.

---

### K. Looking back

**24. What's the biggest limitation?**

**Say:**
Generalisation. All my scenes come from my own generator, so I can't claim it
works on real cities. It also can't tell you when it's unsure. And the
symmetry test shows it learned shortcuts that the physics doesn't allow.

**Follow-ups:**

**"Can you rank them?"**
One, no test on real cities. Two, no confidence estimate. Three, it depends on
the order of the sun vectors. Four, the input: 2.5D only, no neighbours, no
scale. Five, it predicts colours instead of hours, with a GAN loss that rewards
looks. Six, no validation set, no ablations.

**"Which would you fix first?"**
The real-city test, because it decides whether the rest is worth doing. Take a
few hundred real blocks with building heights, run Ladybug, and measure.

---

**25. If you did it again, what would you change?**

**Say:**
Five things, in order. First, predict hours, or nine hourly shadow masks,
instead of colours, and only keep the GAN if an experiment shows it helps.
Second, make the sun input order-free, and augment with rotations and mirrors
that move the sun too. Third, a validation set, a learning curve, and tests of
each loss choice. Fourth, a height map at a fixed scale, with a separate
analysis mask and the neighbouring buildings included. Fifth, a test on real
cities, and a confidence for each pixel.

---

**26. How does this connect to what you want to do in a PhD?**

**Say (adjust to the position):**
What I care about is fast learned models of physical environments that you can
actually trust. This project showed me three open problems first-hand. Will it
work beyond the data I generated? Can it say when it's unsure? It's confident
everywhere right now. And how do you build the physics into the model, so it
doesn't learn shortcuts like the order of the sun vectors?

Pick the one that fits the group: probability and uncertainty for KTH, 3D
representations for MPI and DFKI, agents and generated environments for
Chalmers, choosing and sizing models for Lund and TU Delft TPM.

---

**27. What was the hardest part?**

**Say (use your own experience if you remember it):**
Making sure training and the live tool see exactly the same input. Small
mismatches, like the masking bug, the slot order, the resizing method or the
colour palette, don't crash anything. They just quietly make the results
worse. The other hard part was evaluating honestly once I had a demo that
looked good. My early estimate was 80%. When I measured it properly, it was 59%
exactly right and 91% within an hour, and I learned much more from that.

---

**28. What did you learn?**

**Say:**
That a model that looks right isn't necessarily right. The outputs looked
convincing long before I measured them. Good evaluation, with baselines,
breakdowns and the symmetry test, told me more than the training curves ever
did. And the physics, what the answer depends on and what it doesn't, should
shape the design of the model.

---

**29. What's new compared with pix2pix?**

**Say:**
Honestly, the method isn't new. What's different is the problem setup and the
conditioning. I feed in a physical condition, the sun over a whole day,
through FiLM at 13 levels and into the discriminator too. So one model covers
any place and date, instead of one model per sun position. The evaluation, and
finding where it breaks, is the part I'd build research on.

---

**30. Quick-fire facts**

- Input: 256 × 256 × 3, scaled to −1..1.
- Sun input: 36 numbers = 9 × 3 + 9.
- Sun embedding: 128 numbers.
- Generator: 60.1 million parameters.
- Encoder: 8 steps down, 256 to 1.
- FiLM: 13 layers (enc2–7, dec1–7).
- Normalisation: instance norm, scale and shift come from FiLM.
- Dropout: 0.5 in dec1–dec3, off when used.
- Discriminator: 30 × 30 patch scores, each sees 70 × 70 pixels.
- Loss: least-squares GAN + 50 × L1. Discriminator loss × 0.5.
- Optimiser: Adam, 1e-4, betas (0.5, 0.999), batch 16, 200 epochs, no decay.
- Released checkpoint: epoch 195, about 49,000 updates.
- Data: 4,000 training, 200 test, Grasshopper + Ladybug.
- Result: 90.8% within 1 hour, mean error 0.56 h, 59.3% exact. Baseline
  53.6%, 2.18 h, 39.8%.

---

**31. Questions that test understanding, not memory**

Answer these out loud. The answers follow from everything above.
- If every building were twice as tall, what should happen to the output?
  Would the model know? (Heights are in metres, fixed mapping, so yes, unless
  they go past 250 m.)
- Move the site from 40°N to 40°S on the same date. What changes in the sun
  vectors and in the answer?
- Why isn't the 1 × 1 bottleneck enough on its own? Why aren't the skip
  connections enough on their own?
- What would happen with L1 only? With the GAN only?
- How could you get a confidence for each pixel without retraining? (Monte
  Carlo dropout.) With retraining? (Several models; a classification output;
  a probabilistic model such as a diffusion model.)
- How would you check the model isn't just memorising your generator's
  building shapes? (Real-city test; hold out a whole building category.)
- A reviewer says "this is just pix2pix". What do you say?

---

## Part 2: Eddy3D

Note: the Building Simulation 2027 abstract is still under double-blind review.
Refer to it as "a conference abstract under review" without naming the venue or
details beyond your collaborators.

Code excerpts are from the Eddy3D repository:
`Radiance/CMP/SkyExposureCMP.cs` (sky exposure) and
`Radiance.Core/Radiation/SurfaceTempSolver.cs` (surface temperature).

**1. What is Eddy3D, and which two parts did you build?**

Eddy3D is an open-source urban microclimate simulation plugin for Grasshopper.
I built two parts: the Sky Exposure component, and the surface temperature
solver based on the frequency-domain admittance method (a C# port of the
SurfaceTemps Python method, extended with vegetation evaporative cooling).
Other parts (for example the UTCI fixes and the unit test maintenance) were
done by others; don't present them as yours.

**2. What is sky exposure, and why does it matter?**

How much of the sky each point on the site can see: 0 means fully obstructed,
1 means fully open sky. It determines how much radiation the point exchanges
with the sky, and it is an input for mean radiant temperature (MRT) and the
thermal comfort index UTCI. In the surface temperature solver below, it also
scales how much sun and sky radiation a surface receives.

**3. How did you build it, and why is it 90× faster?**

From each point, 145 rays are cast toward the sky, one per patch of the
Tregenza sky subdivision (8 rings of patches from the horizon to the zenith).
The result is the fraction of rays that hit nothing.

Building the 145 directions:

```csharp
// Tregenza 145 sky directions: 8 elevation rows with decreasing azimuth counts + a zenith cap.
int[] rowCount = { 30, 30, 24, 24, 18, 12, 6, 1 };     // patches per ring, sums to 145
double[] rowElev = { 6, 18, 30, 42, 54, 66, 78, 90 };  // elevation of each ring (degrees)
const double deg2rad = Math.PI / 180.0;

var skyDirs = new Vector3d[145];
var idx = 0;
for (var r = 0; r < rowCount.Length; r++)
{
    var n = rowCount[r];
    var elev = rowElev[r] * deg2rad;
    var sinE = Math.Sin(elev);
    var cosE = Math.Cos(elev);
    if (n == 1)
    {
        skyDirs[idx++] = new Vector3d(0, 0, 1);            // straight up
    }
    else
    {
        for (var i = 0; i < n; i++)
        {
            var az = 2.0 * Math.PI * i / n;                // evenly spaced around the ring
            skyDirs[idx++] = new Vector3d(cosE * Math.Cos(az), cosE * Math.Sin(az), sinE);
        }
    }
}
```

All context geometry (meshes, Breps, surfaces) is first merged into one mesh,
so each ray is tested against a single object:

```csharp
var combined = new Mesh();
foreach (var geom in contextGeo)
{
    switch (geom)
    {
        case Mesh m:
            combined.Append(m);
            break;
        case Brep b:
            var bm = Mesh.CreateFromBrep(b, MeshingParameters.FastRenderMesh);
            if (bm != null) foreach (var x in bm) combined.Append(x);
            break;
        case Surface s:
            var sm = Mesh.CreateFromBrep(s.ToBrep(), MeshingParameters.FastRenderMesh);
            if (sm != null) foreach (var x in sm) combined.Append(x);
            break;
    }
}
```

Then every point is processed in parallel:

```csharp
Parallel.For(0, nPts, i =>                       // points are split across CPU cores
{
    var origin = new Point3d(ptsX[i], ptsY[i], ptsZ[i] + cOffset);  // 0.918 m above the ground
    var hits = 0;
    for (var d = 0; d < cNDirs; d++)
    {
        var ray = new Ray3d(origin, cDirs[d]);
        var t = Intersection.MeshRay(cMesh, ray);  // t >= 0 means the ray hit something
        if (t >= 0) hits++;
    }
    result[i] = 1.0 - (double)hits / cNDirs;     // fraction of open sky
});
```

Why it is fast: a fixed, small set of 145 directions; one merged mesh instead
of many objects; point coordinates copied into plain arrays so the parallel
loop can read them safely; and `Parallel.For` across all cores. On a grid of
18,225 points it is about 90× faster than the Ladybug sky exposure workflow used
before.

Rays start 0.918 m above each ground point. This is the default, and users can
change it. The question is how a person feels outdoors, so the rays start at
about the middle height of a standing person rather than at the ground.

**4. How did you check the results are correct?**

Point-by-point comparison with reference simulations: sky exposure MAE 0.0044;
the downstream UTCI differs by 0.0085 °C on average; correlation 0.9998. In
other words, 90× faster with practically the same results.

**5. What problem does the surface temperature solver solve?**

Many simplified models assume ground temperature equals air temperature, so
the difference between materials (grass, stone, asphalt) disappears. Others
include materials but ignore the heat that plants remove through
transpiration, so grass comes out hotter than paving, which misleads design.
The solver handles both: layered materials with their own thermal properties,
and evaporative cooling for vegetated surfaces.

The per-point pipeline (`SurfaceTempSolver.ComputeAnnual`), over all 8,760
hours of the weather file:

1. Split the sun from the EPW file into direct, sky-diffuse and
   ground-reflected parts for the surface's tilt and orientation
2. Scale direct and sky radiation by the sky exposure from question 3
3. Compute the sol-air temperature (air temperature corrected for absorbed sun
   and for long-wave exchange with the sky)
4. For vegetation, subtract evaporative cooling (latent heat)
5. Run the admittance solver to get the surface temperature for every hour

**6. What is the frequency-domain admittance method?**

The idea: any yearly temperature curve can be written as a sum of sine waves
of different periods (FFT). A material layer damps and delays each sine wave
by an amount that depends only on its period and the material. So: split the
driving temperature into sine waves, apply each wave's damping and delay,
and add the waves back together (inverse FFT).

The sol-air temperature that drives the surface:

```csharp
// (hc·T_air + hr·T_radiant + α·Q_sol − LE) / (hc + hr)
for (var i = 0; i < result.Length; i++)
{
    var he = hc[i] + hr;                         // convective + radiative coefficient
    var le = latentHeatFlux?[i] ?? 0.0;          // evaporative cooling, 0 for hard surfaces
    result[i] = (hc[i] * tAir[i] + hr * tRadiant[i] + alpha * qSol[i] - le) / he;
}
```

How one material layer damps one sine wave (a 2×2 complex transfer matrix):

```csharp
private static (Complex m11, Complex m12, Complex m21, Complex m22) LayerMatrix(
    SurfaceTempLayer layer, double period)
{
    var lam = layer.Conductivity;
    var alpha = lam / (layer.Density * layer.SpecificHeat);   // thermal diffusivity
    var omega = 2.0 * Math.PI / period;                       // frequency of this wave
    var p = Complex.Sqrt(new Complex(0, omega / alpha));
    var pL = p * layer.Thickness;

    var z1 = Complex.Cosh(pL);
    var z2 = Complex.Sinh(pL) / (lam * p);
    var z3 = lam * p * Complex.Sinh(pL);
    return (z1, z2, z3, z1);
}
```

Layers are stacked by multiplying their matrices (`TotalMatrix`). From the
total matrix, each wave gets one complex factor `h[k]`: its size is the
damping, its angle is the delay.

```csharp
for (var k = 1; k <= half; k++)
{
    var periodK = totalPeriod / k;                 // k-th harmonic: one year / k
    var m = TotalMatrix(assembly, periodK);
    var hk = Complex.One / (Complex.One + m.m11 * rSo / m.m12);
    if (k == half) hk = new Complex(hk.Real, 0.0); // the highest-frequency bin must be real
    h[k] = hk;
}
```

The solve itself:

```csharp
public static double[] SolveSurfaceTemperature(
    IReadOnlyList<double> tDriving, SurfaceTempAssembly assembly, double tInternal)
{
    var n = tDriving.Count;                          // 8760 hours
    const double dt = 3600.0;                        // one hour in seconds
    var half = n / 2;

    var tMean = tDriving.Average();
    var buffer = new double[2 * (half + 1)];
    for (var i = 0; i < n; i++) buffer[i] = tDriving[i] - tMean;   // only the fluctuation

    Fourier.ForwardReal(buffer, n, FourierOptions.AsymmetricScaling);  // time -> sine waves

    var h = TransferFunction(assembly, n, dt);       // cached per material
    for (var k = 0; k <= half; k++)                  // damp and delay every wave
    {
        var bin = new Complex(buffer[2 * k], buffer[2 * k + 1]) * h[k];
        buffer[2 * k] = bin.Real;
        buffer[2 * k + 1] = bin.Imaginary;
    }

    Fourier.InverseReal(buffer, n, FourierOptions.AsymmetricScaling);  // sine waves -> time

    var u = SteadyStateU(assembly);                  // the mean is handled separately (steady state)
    var tSoMean = tMean - u * (tMean - tInternal) * assembly.RSo;

    var result = new double[n];
    for (var i = 0; i < n; i++) result[i] = tSoMean + buffer[i];
    return result;
}
```

Why it is fast: one FFT, one multiplication per frequency and one inverse FFT
per point, instead of stepping through 8,760 hours in small time steps as a
finite-difference model would. Two caches make it faster still: the transfer
function depends only on the material, so every point with the same material
reuses it; and the weather-derived series (sky temperature, convection
coefficient, sun position) are computed once per weather file, not per point.

**7. What problems did you find along the way?**

A sign inconsistency in a published formula, which I corrected. Also,
vegetated surfaces initially came out unrealistically hot, so I added a latent
heat term (Priestley-Taylor method) for the heat removed by transpiration:

```csharp
// LE = α · Δ/(Δ+γ) · Rn, clamped to >= 0 (daytime cooling only)
for (var i = 0; i < result.Length; i++)
{
    var delta = SaturationVaporPressureSlope(tAir[i]);   // depends on air temperature
    result[i] = Math.Max(0.0, alphaPt * (delta / (delta + gamma)) * netRadiation[i]);
}
```

Then it is scaled by each material's moisture availability (0 for paving, so
hard surfaces are unaffected):

```csharp
if (assembly.MoistureAvailability > 0.0)
{
    var netRadiation = new double[n];
    for (var i = 0; i < n; i++) netRadiation[i] = assembly.Absorptivity * qSol[i];

    var lePotential = SurfaceTempSolar.PriestleyTaylorLatentHeat(netRadiation, weather.DryBulbTemp);
    latentHeatFlux = new double[n];
    for (var i = 0; i < n; i++) latentHeatFlux[i] = assembly.MoistureAvailability * lePotential[i];
}
```

Priestley-Taylor only needs net radiation and air temperature, no wind or
humidity, which fits the data already available in the pipeline.

**8. What are the limitations?**

- Sky exposure counts all 145 directions equally. Tregenza patches have similar
  solid angles, so this approximates the fraction of visible sky, but it is not
  the cosine-weighted sky view factor used in radiation exchange
- The surface solver uses the yearly sky exposure as a stand-in for hour-by-hour
  sun visibility: it does not test whether the sun itself is blocked at each
  hour, as the original Python version does
- Ground-reflected radiation is not reduced by surrounding buildings
- No reflected radiation between surfaces (single-bounce radiosity)
- The comparison with Outdoor+, EnergyPlus, Radiance and Ladybug in the abstract
  is an expected result; the full paper is still being written, so don't
  present it as a finished finding

---

## Part 3: Follow-up questions to think about first

- Why rays for sky exposure and not another method? Why the 145 Tregenza directions and not more?
- Why is the surface temperature solved over the whole year (8,760 hours) and not one day?
  - Answer: the reference method (SurfaceTemps) is formulated over the whole year, so the port keeps that. Once the 8,760-hour series exists, any single day or hour is just a slice of it, so nothing is lost.
