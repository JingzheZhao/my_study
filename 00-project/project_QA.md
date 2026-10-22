# Project Q&A: InstantSim and Eddy3D

How to use: don't memorize the wording. Read one question, close the file, and
answer out loud in your own words, then open it and check what you missed. In
Part 1 every question has a **short answer** (what you say first, 30 to 60
seconds), the **code** it refers to, and a **follow-up chain**: the questions a
PhD interviewer is likely to ask next, each with an answer. Practise the chains,
not just the first answer; that is where interviews are decided.

Facts only you knew are confirmed in Part 0. Anything still unknown is marked
as not recorded: say so in the room, never guess.

InstantSim code is from the repository: `training/train_custom_solar.py`
(model, data, training), `backend/app/models/networks.py` (the model used for
inference and evaluation), `training/evaluate_model.py` and
`training/probe_symmetry.py` (evaluation), `backend/app/core/inference.py` and
`backend/frontend/index.html` (deployment).

---

## Part 0: Confirmed facts (updated 2026-10-20)

Always use these numbers in interviews. README and docs match them.

1. **Accuracy** on 200 held-out test scenes (`training/evaluate_model.py`):
   - Exact hour class **59.3%**, within ±1 hour **90.8%**, MAE **0.56 h**
   - No-shadow baseline (every pixel gets full daylight hours): 39.8%, 53.6%, 2.18 h
   - 49 training scenes: 70.3%, 95.4%, 0.38 h
   - Near class boundaries (76% of site pixels): 49% exact, 88% within ±1 h.
     Inside uniform regions: 91% exact, 99% within ±1 h
   - High-rise scenes MAE 0.73 h, low-rise 0.40 h. Ground MAE 0.50 h, roofs 0.92 h

   These replace 57.1% / 90.9% / 0.58 h from 2026-10-19 (see item 10). The 80%
   in the application letters was an early rough estimate (images compared with
   ChatGPT, not code). If asked: you later evaluated systematically on a
   held-out test set, and these are the results.
2. **FiLM is used in 13 blocks**: enc2–enc7 (6) and dec1–dec7 (7). enc1 and
   enc8 have no normalization and no FiLM.
3. **No data augmentation.**
4. **Checkpoints** every 5 epochs, no validation set. The released model is the
   epoch-195 checkpoint (0-indexed): the original script only saved when
   `epoch % 5 == 0`. The script now also saves after the last epoch.
5. **10 classes, 0 to 9 hours.** "Nine classes" in the letters is wrong.
6. **Label = direct sun hours** between 9:00 and 17:00 (9 hourly sun
   positions), the time people are mostly outdoors.
7. **LSGAN loss, L1 weight 50.** No ablation was run for either.
8. **InstanceNorm only**, no BatchNorm.
9. **Data**: in Grasshopper a site is randomly subdivided and each plot randomly
   extruded; height, density and footprint shape are sampled by category. Say
   "parametric random generation", not "procedural".
10. **Bugs found and fixed on 2026-10-20** (good story for "tell me about a bug"):
    - The inference/evaluation copy of `SunConditionEncoder` forgot to zero the
      embeddings of hours without sun; the training code did. Only short days
      were affected: their exact accuracy went from 41–49% to 58–65% after the
      fix, and overall from 57.1% to 59.3%.
    - The browser put the sun vectors in fixed hour slots (slot 0 = 9:00, a zero
      vector if the sun is down), but the training data packs the daylight
      hours at the front. Fixed to match the training layout.
11. **Symmetry probe** (`training/probe_symmetry.py`, 50 test scenes): same
    scene with the 9 sun vectors in reverse order (label unchanged): exact
    accuracy drops from 59.8% to 29.0%. Mirrored scenes with mirrored sun: 50.2%
    (east-west) and 52.5% (north-south). Meaning: the model has partly learned
    the geometry, but it depends on the order of the inputs even though the
    physics does not. See questions 14 and 31.
12. **How the height map is made** (Grasshopper script in `data-generator.gh`,
    the one that sends to `/predict`): it is a 512×512 screenshot of the Rhino
    Top view in Rendered display mode, not a raycast. Before the screenshot,
    every mesh gets vertex colours:
    - ground 240, a background plane 255 under everything
    - buildings: gray = 195 − 195 × z / 250, with z the vertex height in metres
      clipped to 0..250 m. So 0 m → 195, 125 m → 98, 250 m and above → 0
    - the camera zooms to a square of 1.2 × the longer side of the bounding box
      of ground + buildings, so the metres per pixel depend on the size of the
      site

13. **Height colouring, confirmed 2026-10-21.** In Grasshopper: vertex z →
    Remap Numbers from 0..250 m to 60..255 → Gradient (white at 0, black at 255)
    → Mesh Colours. Ground is a flat colour of 240. So a building at 0 m gets
    gray 255 − 60 = 195 and a building at 250 m gets 0, the same mapping as the
    Python capture script. The gradient's parameter runs 0..255, but buildings
    only use 60..255 of it, so their grays are 0..195. Why 60: it leaves a gap
    between the lowest building (195) and the ground (240), so even a very low
    building is never confused with ground. 250 m covers most sites.
    **Measured in the training images:** the Rendered display mode applies a
    gamma curve. The dark values in high-rise scenes are exactly
    255 × (c / 255)^(1/2.2) for consecutive integer grays c = 0, 1, 2, 3, …
    (observed 0, 20, 28, 34, 39, 43, 46, 50, 53, …; 89% of pixels below 120 fall
    on this curve). So the stored pixel is a gamma-encoded, not linear,
    function of height: tall buildings are spread over many levels, low
    buildings are squeezed into few. In numbers: 0 m (gray 195) is stored as
    about 226, 50 m (156) as about 204, 125 m (98) as about 165, 250 m and
    above as 0. The ground stays at 240. Values between 226 and 240, and between
    240 and 255, are anti-aliased edges (building-ground and ground-background).
    So the low-rise test scenes top out at about 50 m, and high-rise scenes
    contain towers at or above 250 m, which are clipped.
    **Training vs. use:** training images were exported from the Rhino view;
    at inference the Python script captures the view. The last inference input
    saved by the backend (`debug_outputs/debug_ai_input.png`) shows the same
    encoding as the training images: ground 240, buildings on the same gamma
    curve (0, 20, 28, 34, …), lowest buildings around 226. So the two pipelines
    agree; the model is not seeing a different encoding at inference.

**Confirmation list (all answered 2026-10-21)**
- a. Answered 2026-10-21 (see item 13).
- b. Answered 2026-10-21: each training site starts as a 1000 m × 1000 m
  square, scaled about its centre by 0.6–1.25 in x and 0.6–1.3 in y, so sites
  are 600–1250 m by 600–1300 m. The camera frames 1.2 × the longer side, so
  the captured square is 720–1560 m wide: 1.4–3.0 m per pixel in the 512 px
  image and 2.8–6.1 m per pixel at the model's 256 px, a factor of about 2.2
  between scenes. The model is never told which (see question 4).
- c. Answered 2026-10-21: latitude uniformly random from 35°S to 65°N,
  longitude −180° to 180°, date random over the whole year, hours fixed at
  9:00–17:00. Checked in the training log: on full 9-hour days the sun is
  highest at the 12:00 slot in 90% of scenes, so the window is in local solar
  time (time zone ≈ longitude / 15, no daylight saving), the same convention
  the browser uses.
- d. Answered 2026-10-21: Ladybug grid size 5 m (the sites are large),
  offset 0.2 m above the surface, time step 1 per hour.
- e. Answered 2026-10-21: the original test set was generated separately from
  the training set, with its own sites and sun conditions. Those files are no
  longer on disk. The 200-scene `test/` folder that the reported numbers were
  computed on is a different set: 199 of its 200 scenes have exactly the same
  site outline and sun vectors as a training scene (training ids 1427–2399),
  with different buildings. So the current numbers measure new building layouts
  on sites and sun conditions seen in training; they are not the original
  independent test. Its mix: 51 low-rise (about 50 m), 49 mid-rise, 100
  high-rise. A 600-scene set in `E:\InstantSim\dataset\test_A` / `test_B`
  (200 per height category) has no sun-vector log, so it can't be evaluated.
- f. Answered: trained on your own PC, Intel Core i7 (9th gen), RTX 3060 Ti,
  32 GB RAM. Training time not recorded; say so rather than guess.
- g. Answered: Ladybug takes roughly tens of seconds to a minute per scene at a
  5 m grid. "~100 ms" was an estimate from the result appearing instantly in
  use; the measured forward pass is about 110 ms on a 2-core cloud CPU and
  would be faster on the 3060 Ti. End-to-end time was never measured.
- h. Not remembered. Don't describe earlier versions in an interview unless
  you remember them.

**About AI-written code.** Interviewers assume every design choice is yours.
When asked why something is done a certain way, answer with the reasoning in
this file. If asked directly whether you used AI to write code, say yes, you
used AI coding tools, and you understand and can explain every design decision.
This document is how you get there.

---

## Part 1: InstantSim

### A. Problem and motivation

**1. Describe the project in one or two sentences.**

Short answer: computing sun exposure for an urban design takes a physics
simulation that runs for minutes, so designers rarely check it while they
iterate. I trained a conditional image-to-image network that takes a height map
of the buildings and the sun directions for a day and returns a direct sun hours
map in about 0.1 s, with 91% of pixels within one hour of the simulation.

Follow-ups:
- *What is "direct sun hours" exactly?* For every point on the ground or a roof,
  the number of hours (here among 9:00, 10:00, …, 17:00) at which the sun is
  above the horizon and the straight line from the point to the sun is not
  blocked by a building. Ladybug computes it by casting one ray per point per
  sun position and counting the rays that escape. Settings used: test points on
  a 5 m grid, 0.2 m above the ground or roof, one sun position per hour.
- *9:00 to 17:00 is 8 hours. How can a point get 9?* Ladybug samples the sun at
  the 9 hour marks (9:00, 10:00, …, 17:00) and counts each unblocked sample as
  one hour. So the label is a count of sampled instants, an approximation of
  the continuous time in sun; a shadow edge that passes a point at 10:20 is
  rounded to whole samples.
- *Why 0.2 m above the surface, when Eddy3D uses 0.918 m?* Different questions.
  Sun-hours guidelines judge sun on the ground and on open space, so the points
  sit just above the surface; the small offset only keeps the ray from hitting
  the surface it starts on. Eddy3D's sky exposure is about how a standing person
  feels, so it starts at body mid-height.
- *Why does a designer care?* Sun on public space, playgrounds and courtyards
  drives outdoor comfort and is in many planning guidelines (for example
  minimum sun hours on open space). It is checked at the end of design today
  because the simulation is slow.
- *Why 9:00 to 17:00 and not sunrise to sunset?* That is when people are mostly
  outdoors, and it fixes the output range to 0..9, which keeps the label space
  the same for every day of the year.
- *Is this a research contribution or an application?* Honest answer: the
  network is a known recipe (pix2pix-style conditional GAN). The contribution is
  the formulation (sun directions as a condition so one model covers all
  locations and dates), the data pipeline, and the end-to-end tool. The
  evaluation in Part 0 also shows where the recipe falls short, which is what I
  want to work on next.

**2. Why learn a surrogate at all? Why not make the ray tracing faster?**

Short answer: a fair alternative. GPU ray tracing would also be fast and exact
for this particular quantity. I chose learning because the same pipeline
extends to quantities with no fast exact solver (wind, surface temperature,
thermal comfort), where simulation takes hours. Sun hours was the right first
target because the ground truth is cheap and exact, so I could study the
surrogate itself.

Follow-ups:
- *So for sun hours specifically, is the network worth it?* For accuracy, no:
  a GPU ray tracer is exact. Its value here is as a test bed with perfect labels
  and as a fast, differentiable stand-in that can sit inside an optimisation
  or generation loop.
- *What does "differentiable" buy you?* Gradients of sun hours with respect to
  the height map, so a generator or optimiser can be steered toward more sun on
  open space without black-box search.
- *What would change for wind?* The physics is non-local and depends on wind
  direction and speed instead of sun vectors; the output becomes continuous
  velocity fields; labels come from CFD (hours per case), so the dataset is much
  smaller and generalisation becomes the main problem. The conditioning
  mechanism (FiLM with the inflow direction) carries over.

**3. Where does the speed-up come from, and how did you measure it?**

Short answer: Ladybug casts rays for every grid point and every hour; the
network does one forward pass of a fixed cost regardless of how many buildings
there are. Ladybug takes roughly tens of seconds to a minute per scene (5 m
grid, 9 sun positions); the network's forward pass takes about 110 ms on a
2-core CPU, so a few hundred times faster. Honest detail: the "~100 ms" in the
README was an estimate from the result appearing instantly; end-to-end time
was never measured.

Follow-ups:
- *How many operations is the forward pass?* The generator has 60.1 million
  parameters. Cost is fixed by the image size, not by scene complexity.
- *What dominates end-to-end latency?* The viewport screenshot in Rhino, the
  upload and the network forward pass; the post-processing (colour matching on
  65k pixels) is small. Not measured; it would be the first thing to measure.
- *How would you make it faster?* GPU inference, half precision, a smaller
  network (the 1×1 bottleneck and 512-channel layers are oversized for this
  task), or distillation into a smaller student.

### B. Input 1: the height map

**4. Why represent the scene as a height map?**

Short answer: image-to-image networks need a regular grid. A top-down height map
puts the whole scene into one channel at a fixed resolution, it is cheap to
make (a screenshot of the Rhino Top view with heights painted as gray values),
and it contains exactly what casts shadows on the ground: where the buildings
are and how tall they are.

How the height map is made (Grasshopper script, `create_colored_meshes` and
`action_send`):

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

# a huge plane with colour 255, 50 m below the lowest geometry, is the background

# camera: Top view, Rendered mode, zoom to a square around the scene
max_dim = max(dx, dy)                         # longer side of the bounding box
half_side = (max_dim * ZOOM_FACTOR) / 2.0     # ZOOM_FACTOR = 1.2, a 20% margin
vp.ZoomBoundingBox(target_bbox)

capture = Rhino.Display.ViewCapture()
capture.Width = 512
capture.Height = 512
bitmap = capture.CaptureToBitmap(view)        # this PNG is sent to /predict
```

How the backend reads it (`inference.py`):

```python
def _build_valid_mask_from_input(self, input_bgr):
    # Background = 255 (outside the site) -> invalid
    # Ground = 240, buildings < 240 (darker = taller) -> valid
    gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    valid = (gray != 255)
    kernel = np.ones((3, 3), np.uint8)
    valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    return valid
```

Follow-ups:
- *What can a height map not represent?* Anything with more than one height per
  pixel: overhangs, arcades, bridges, tree canopies with space under them,
  pilotis. It also can't say what a facade looks like; for ground-level sun
  hours that rarely matters, for facade sun hours it does.
- *Why a viewport screenshot?* It is the simplest way to get an image out of
  Rhino: colour the meshes, look from the top, capture. Seen from the top, a
  flat roof shows only its own vertices, so the pixel gets the roof height.
- *What is wrong with a screenshot?* The Rendered mode applies lighting and
  anti-aliasing, so the pixel values are not exactly the colours that were
  assigned. Edges become blends of neighbouring values (the data contains values
  like 241–251 between ground and background that no object was given), and
  lighting can shift the grays. A sloped roof also interpolates colours between
  its vertices. The image depends on display settings, not only on geometry.
- *What would be better?* Cast one ray straight down per pixel (or read a depth
  buffer) and write the hit height as a number, preferably in a 16-bit or float
  image so heights are not rounded to 196 levels. Then the input depends only on
  the geometry and is the same in training and use.
- *How fine is the height resolution?* 250 m over 195 gray levels: about 1.3 m
  per level. Enough for shadows of buildings; it can't distinguish a 1 m wall
  from flat ground.
- *Why three identical channels?* So the input matches the 3-channel pix2pix
  layout. It wastes two channels. A better input would be one channel for height
  plus one channel for the analysis mask (see the next question).
- *Is "255 = outside the site" a good encoding?* It mixes two different things:
  "not part of the analysis" and "empty space". Shadows from neighbouring
  buildings outside the site can't be represented, but in a real city they
  matter. Better: a geometry channel that includes the context buildings and a
  separate binary mask for where to evaluate.
- *How is height mapped to a gray value?* Linearly: 0 m is 195, 250 m is 0,
  darker is taller; ground is 240 and the background 255. The gap between 195
  and 240 separates the lowest building from the ground.
- *Why use the vertex z and not the height above the ground?* Simpler, but it
  assumes the ground is at z = 0. On a sloped site, or a model placed at an
  elevation, every building would be read as taller or shorter than it is.
- *Does the network know absolute heights?* Yes, as long as the mapping is the
  same for every scene, which it is in this script (fixed 0–250 m, not
  normalised per scene). That matters because shadow length depends on absolute
  height (L = H / tan(altitude)). Training images and inference inputs use the
  same mapping (checked, Part 0 item 13).
- *What if a building is taller than 250 m?* It is clipped to 0, so the model
  sees a 250 m building and under-predicts its shadow. High-rise test scenes
  have 1,000–4,500 pixels at exactly 0 per scene, which may be clipped towers.
- *Does the network know how many metres a pixel is?* No, and this is a real
  flaw. The camera zooms to 1.2 × the longer side of the scene. Training sites
  were 600–1300 m on their longer side, so at the model's 256 px one pixel is
  anywhere from 2.8 m (600 m site) to 6.1 m (1300 m site), a factor of 2.2. Heights are in metres but distances are in pixels, so the same building
  should cast a shadow twice as many pixels long on the small site. The model
  can't know which case it is in. This is true in the training data too: the
  training sites have different sizes, so the same building with the same sun
  casts shadows of different pixel lengths in different training scenes. The
  model can only learn an average scale, which puts a floor under the error,
  most of all for long shadows (low sun, tall buildings), where a scale error
  moves the shadow edge by many pixels. It is one plausible reason high-rise
  scenes are the hardest (MAE 0.73 h vs 0.40 h), not yet tested.
- *Why did you randomise the site size?* So the model wouldn't only know
  1000 m × 1000 m sites and would generalise to other sizes. The intent was
  right: varying the data is how you avoid overfitting to one setting. In
  hindsight the capture undid it. Every site is zoomed to fill the image, so
  the model never sees "a bigger site"; it sees the same-sized image at an
  unknown scale. The variation I meant as diversity became ambiguity. Done
  properly, the site size would vary at a fixed metres per pixel (a bigger site
  covers more of the image, or is tiled), or the scale would be given to the
  model as an input.
- *How would you fix the scale problem?* Either a fixed metres-per-pixel (always
  capture the same physical area, for example a 300 m square, and pad), or give
  the scale to the model as an extra condition, or encode height in pixel units
  (height divided by the pixel size), which makes the geometry scale-free:
  then shadow length in pixels = height in pixels / tan(altitude) whatever the
  site size.

**5. What if the user's image isn't 256×256?**

Short answer: the backend resizes every input to 256×256 with nearest-neighbour
interpolation, exactly as in training, and upsamples the output to 512×512 for
display.

```python
# inference.py
img_input_bgr_256 = cv2.resize(img_input_bgr, (256, 256),
                               interpolation=cv2.INTER_NEAREST)

# train_custom_solar.py, the same for the training input
transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
```

Follow-ups:
- *Why nearest and not bilinear for the height map?* Bilinear would invent
  heights that don't exist at building edges (a 30 m roof next to 0 m ground
  becomes a 15 m pixel), which the network would read as a small building.
- *What if the image isn't square?* It is stretched, which distorts building
  footprints and shadow lengths. The Grasshopper side always makes a square
  image, but the API doesn't check. Better: pad to a square, or crop and tile.
- *Can the network run at 512 or 1024 directly?* It is fully convolutional, so
  technically yes for multiples of 256, but with 8 down-sampling steps the
  bottleneck becomes 2×2 or 4×4 and the network never saw those statistics in
  training, and shadow lengths in pixels change. So in practice no without
  retraining.
- *What about a larger site?* Either a coarser pixel (less detail) or tiling.
  Tiling needs overlap, because a shadow can cross a tile border: a tile needs
  context of at least the longest shadow, H / tan(altitude), from its
  neighbours.
- *Why 256 at all?* It is the standard pix2pix size and fits batch 16 in GPU
  memory. The cost is resolution: training sites are 600–1300 m, so at 256 px a
  pixel is 2.8–6.1 m. A 20 m wide building is only 3–7 pixels across, and a
  15 m street 2–5 pixels, so courtyards and narrow streets, exactly where sun
  matters for people, are a few pixels wide. This is why one-pixel errors at
  shadow edges cost so much in exact accuracy.
- *Then is the model suited to a single building or a small plaza?* Not as
  trained. It learned neighbourhood-scale sites of roughly 1 km; a 100 m site
  would be stretched to fill the image at 0.5 m per pixel, a scale it never
  saw.
- *Does the 512 px capture always show exactly the square you zoomed to?*
  `ZoomBoundingBox` fits the square into the viewport on screen; if the
  viewport isn't square, the 512×512 capture can show a slightly different area
  than intended, which changes the scale again. Worth checking.

**6. Is the site orientation important?**

Short answer: yes. The sun vectors are in world coordinates, so the image must
be north-up and aligned with the world x and y axes, as in training. Rotating
the site in the image without rotating the sun vectors gives wrong shadows.

Follow-ups:
- *How did you check which image axis is which world axis?* With the symmetry
  probe: flipping the image top-bottom and negating the y component of every
  sun vector keeps accuracy close to the original (52.5% vs 59.8% exact),
  consistent with image rows running along world y.
- *Could you make the model rotation-invariant?* The physics is equivariant:
  rotate the scene and the sun together and the label rotates. That can be
  built in by data augmentation (rotate image and vectors together) or by
  rotating everything into a canonical frame (for example sun azimuth at noon
  pointing up) before the network and rotating the output back.

### C. Input 2: the sun conditions

**7. Why are the sun conditions 36 numbers?**

Short answer: 9 sun positions (one per hour from 9:00 to 17:00) × 3 components
of the sun direction = 27, plus 9 flags that say whether each slot holds a real
sun position (1) or is empty because the sun is down (0).

```python
# SolarDataset: the data log stores only the hours with sun, in time order
vecs = entry.get('vectors', [])                 # list of [x, y, z], up to 9

raw_vecs = np.zeros((9, 3), dtype=np.float32)   # unused slots stay zero
mask = np.zeros((9,), dtype=np.float32)

num_to_fill = min(len(vecs), 9)
if num_to_fill > 0:
    raw_vecs[:num_to_fill] = np.array(vecs[:num_to_fill], dtype=np.float32)
    mask[:num_to_fill] = 1.0                    # packed at the front

vec36 = np.concatenate([raw_vecs.flatten(), mask], axis=0)   # 27 + 9 = 36
```

A real entry from the test log (8 daylight hours; 17:00 is missing):

```
{"id": "0000", "vectors": [[-0.548, 0.7248, -0.4175], [-0.3194, 0.8071, -0.4965],
 ..., [0.9249, 0.3723, -0.0771]]}
```

Follow-ups:
- *What does one vector mean?* A unit vector pointing from the sun toward the
  ground (Ladybug convention), so z is negative. Altitude = asin(−z): the first
  vector above has z = −0.4175, an altitude of about 25°.
- *Why sun vectors and not latitude, longitude, date and time?* Shadows depend
  only on where the sun is, not on why it is there. Vectors are the physical
  cause; latitude and date are an indirect description the network would have
  to convert with astronomy it has no reason to learn well. Different places
  and dates with the same sun path give the same label, and vectors make that
  explicit.
- *Why the mask, if empty slots are already zero vectors?* A zero vector is not
  a valid direction, but the network can't know that unless told; after the
  first linear layer a zero input becomes the bias, which is not zero. The mask
  removes that ambiguity and lets the model zero those slots (question 9).
- *Why is slot 0 not always 9:00?* Because the log keeps only daylight hours and
  packs them at the front. On a short day slot 0 may be 10:00. So slot index is
  "the k-th daylight hour", not a clock time. That is a weakness: the meaning of
  a slot changes with the day length. The network can still tell from the
  vector itself (morning sun points west) but it has to learn that.
- *Where do the vectors come from at inference time?* The browser computes the
  sun position with SunCalc for the chosen place and date at each hour from 9 to
  17 and converts it to the Ladybug convention. Training used Ladybug's sun
  path. The two algorithms differ by small fractions of a degree, which is far
  below what the model resolves.
- *What time zone does "9:00" use?* In the browser, `round(lon / 15)` hours from
  UTC, i.e. solar-ish time without daylight saving. A city whose legal time zone
  differs from its longitude (Spain, western China) gets a window shifted by up
  to an hour or more. The training data uses the same convention: on full days
  the sun is highest at the 12:00 slot in 90% of scenes, i.e. local solar time.
  So training and use agree, but "9:00–17:00" means solar time, not the
  clock time a user in Madrid or Toronto in summer would expect.
- *Why are 44% of training days shorter than 9 hours?* The window is 3 hours
  before solar noon and 5 after, so in winter at mid and high latitudes the
  sun sets before 17:00. That is why the evening end is missing far more often
  than the morning end (1,386 vs 392 short training days).
- *Why 35°S to 65°N?* It covers most cities by latitude. Note that it is
  asymmetric: nothing south of 35°S (Melbourne, Auckland, Wellington are
  outside), and sampling is uniform in latitude, not weighted by where people
  live.

```javascript
// index.html: data time offset used for the 9 vectors sent to the backend
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
// Same layout as the training data: daylight hours packed at the front,
// unused slots zero-padded at the end
while (sunVectors.length < 27) sunVectors.push(0.0, 0.0, 0.0);
```

- *Why hourly sun positions and not finer?* Ladybug's direct sun hours with a
  time step of 1 counts the hour marks, so the label itself is hourly. Finer
  time steps would give fractional hours and a smoother label; the
  conditioning would need more slots or a set encoder (question 14).

**8. How does a 36-number vector get into a convolutional network?**

Short answer: a small encoder turns it into a 128-number embedding. Each of the
9 vectors goes through the same small MLP, empty hours are zeroed, the 9
results are concatenated in order with the mask and the day length, and a
linear layer fuses them. Every FiLM layer then reads this embedding.

```python
class SunConditionEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.per_vec = nn.Sequential(               # shared by all 9 hours
            nn.Linear(3, emb_dim), nn.LeakyReLU(0.2, True),
            nn.Linear(emb_dim, emb_dim), nn.LeakyReLU(0.2, True),
        )
        fuse_in = 9 * emb_dim + 9 + 1               # 1152 + 9 mask + 1 day length
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

Follow-ups:
- *Why share one MLP across the 9 hours?* Lifting a direction into features is
  the same job for every hour, so sharing means 9× fewer parameters and every
  training example teaches it 9 times. Position information comes from where
  the result sits in the concatenation.
- *Why the day-length feature, when the mask already contains it?* It is
  redundant (the mask sum divided by 9); it gives the fuse layer the number
  directly instead of making it count. A small convenience, not essential.
- *Why 128 dimensions?* A conventional size; 36 inputs don't need more. Not
  tuned. The FiLM layers map 128 to up to 1,024 numbers (γ and β for 512
  channels).
- *Is one global embedding enough?* For sun direction, yes: the condition is the
  same for every pixel. A spatially varying condition (for example different
  materials per pixel) would need a spatial method such as SPADE.

**9. What was the masking bug?**

Short answer: training multiplied the per-hour features by the mask; the copy of
the encoder used for inference and evaluation didn't. On days with fewer than 9
daylight hours the empty slots then fed the MLP's bias output, which the model
had never seen in training. Fixing it raised exact accuracy on short days from
41–49% to 58–65%, and overall from 57.1% to 59.3%.

```python
# backend/app/models/networks.py, after the fix
feat = self.per_vec(v)
# Zero the embeddings of hours without sun, as in training
# (per_vec(0) is not zero because of the biases)
feat = feat * m01.unsqueeze(-1)
```

Follow-ups:
- *How did you find it?* The evaluation broken down by day length showed short
  days much worse than full days, which pointed at the code paths that only
  matter for short days; comparing the training and inference encoders line by
  line found the missing line.
- *Why didn't it break full days?* With 9 daylight hours the mask is all ones,
  so the multiplication changes nothing.
- *How do you prevent this kind of bug?* One model definition imported by both
  training and inference (the training script has its own copy), and a test
  that runs both on the same input and checks the outputs are identical.
- *Was there a second mismatch?* Yes, the browser placed hours in fixed clock
  slots while training packs them at the front. Also fixed; it only affected
  days where the sun is down at 9:00.

### D. Output

**10. What does the network output, and why a colour image?**

Short answer: a 3×256×256 RGB image in Ladybug's legend colours, 10 colours for
0 to 9 hours. Honestly, because the Ladybug output image was the easiest label
to produce and it fits pix2pix directly. The colours are converted back to
hours afterwards by nearest legend colour in Lab space.

```python
self.last = nn.Sequential(
    nn.ReLU(True),
    nn.ConvTranspose2d(128, output_nc, 4, 2, 1),   # 128 -> 256 pixels, 3 channels
    nn.Tanh()                                      # [-1, 1], same scale as the label
)
```

Follow-ups:
- *What is wrong with predicting colours?* The loss works in RGB, not in hours.
  The legend runs blue, purple, red, orange, yellow, so the RGB distance between
  two classes isn't proportional to their difference in hours, and the network
  wastes capacity on 3 channels for a 1-D quantity. Errors land between legend
  colours and have to be snapped back.
- *What would you predict instead?* Three options:
  1. Regression: one channel with hours, L1 loss in hours. Simple and direct.
  2. Classification: 10 channels, cross-entropy per pixel. Gives a probability
     per class, so a built-in uncertainty, but ignores that 3 is closer to 4
     than to 9.
  3. Ordinal regression: predict P(hours > k) for k = 0..8, which respects the
     order. The best fit for a count.
- *Better still?* Use the physics: direct sun hours is a sum over hours of
  "is this pixel lit at hour i". Predict 9 shadow masks, one per sun vector,
  and sum them. Each mask is a simpler problem (a shadow for one sun
  direction), the sum is automatically between 0 and the number of daylight
  hours, the order of the hours no longer matters, and a single shadow mask can
  be supervised and checked on its own.
- *Why Tanh?* The labels are normalised to [−1, 1] (`Normalize(0.5, 0.5)`),
  and Tanh outputs exactly that range.
- *Can the model predict more hours than the sun is up?* In principle yes. In
  practice only 0.7% of pixels on short-day scenes exceed the possible maximum,
  so it has learned the cap. A physics-based output (summing per-hour masks)
  would guarantee it.

**11. Why are labels resized with bilinear interpolation but height maps with
nearest?**

Short answer: that is in the training code: the height map uses nearest to keep
hard edges, the label uses bilinear. In hindsight bilinear on the label is
questionable: averaging two legend colours produces a colour that is not in the
legend, so the network is trained to output in-between colours at class
boundaries.

```python
self.transform_A = transforms.Compose([...,
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST), ...])
self.transform_B = transforms.Compose([...,
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR), ...])
```

Follow-ups:
- *Does it matter much?* Probably a little, at boundaries, which is where most
  errors are. Also the original 512 px Ladybug images already contain blended
  colours: only about 82% of site pixels are exact legend colours. The
  evaluation resizes labels with nearest and snaps every pixel to the nearest
  legend colour, so the metrics are in clean classes.
- *What would you do?* Convert labels to hour indices first, then resize with
  nearest (or average in hours, if predicting hours by regression).

### E. Architecture

**12. Why a U-Net? Walk me through it.**

Short answer: the output is a map of the same size as the input and aligned
with it pixel by pixel. A U-Net compresses the image in 8 steps (256 → 1) to
get global context, expands it back in 8 steps, and skip connections pass every
encoder feature map to the decoder at the same size, so building edges stay
sharp.

```python
def forward(self, x, vec):
    emb = self.cond_enc(vec)                 # sun conditions -> 128 numbers

    e1 = self.enc1(x)                        # 256 -> 128, no norm, no FiLM
    e2 = self.enc2(e1, emb)                  # each DownBlock: LeakyReLU, conv k4 s2, FiLM
    e3 = self.enc3(e2, emb)                  # 64
    e4 = self.enc4(e3, emb)                  # 32
    e5 = self.enc5(e4, emb)                  # 16
    e6 = self.enc6(e5, emb)                  # 8
    e7 = self.enc7(e6, emb)                  # 4 ... 2
    e8 = self.enc8(self.enc8_act(e7))        # 1×1, no norm, no FiLM

    b, c, h, w = e8.size()                   # sun embedding as 512 extra channels
    v = self.bottleneck_fc(emb).view(b, 512, 1, 1).expand(b, 512, h, w)

    d1 = self.dec1(torch.cat([e8, v], 1), emb)
    d1 = torch.cat([d1, e7], 1)              # skip connection
    d2 = self.dec2(d1, emb); d2 = torch.cat([d2, e6], 1)
    d3 = self.dec3(d2, emb); d3 = torch.cat([d3, e5], 1)
    d4 = self.dec4(d3, emb); d4 = torch.cat([d4, e4], 1)
    d5 = self.dec5(d4, emb); d5 = torch.cat([d5, e3], 1)
    d6 = self.dec6(d5, emb); d6 = torch.cat([d6, e2], 1)
    d7 = self.dec7(d6, emb); d7 = torch.cat([d7, e1], 1)
    return self.last(d7)                     # ConvTranspose + Tanh -> 3×256×256
```

Follow-ups:
- *Why does a shadow problem need global context?* Shadow length is
  H / tan(altitude): at 10° altitude a building casts a shadow 5.7 times its
  height, at 30° 1.7 times. The training data includes sun altitudes down to
  0–2°, so a shadow can cross most of the image. A pixel's value can depend on a
  building far away. The 1×1 bottleneck sees the whole image, and the
  receptive field of the deeper layers covers it too.
- *Then what do the skip connections add?* Exact positions. The bottleneck knows
  "there is a tall building to the south-west"; the 128×128 skip from enc1 knows
  exactly which pixels are its edge. The decoder combines both.
- *Why 4×4 convolutions with stride 2 instead of 3×3 plus pooling?* The pix2pix
  design: down-sampling is learned instead of fixed max pooling, and a kernel
  of 4 with stride 2 covers each input pixel evenly. A known side effect of the
  matching transposed convolution is checkerboard artefacts.
- *Why LeakyReLU in the encoder and ReLU in the decoder?* Also from pix2pix and
  DCGAN: leaky units keep a gradient for negative inputs in the encoder; the
  choice in the decoder matters little.
- *Why does enc8 have no normalisation?* It is 1×1: instance normalisation over a
  single pixel subtracts the value from itself and outputs zeros, which would
  erase the bottleneck. enc1 has none either, following pix2pix, which keeps the
  raw input level in the first skip connection.
- *What does dropout do here?* Dropout 0.5 in dec1–dec3 during training, as in
  pix2pix. At inference `model.eval()` turns it off, so the output is
  deterministic. pix2pix kept dropout on at test time as a noise source; turning
  it on and running the model 20 times would give a cheap uncertainty estimate
  (Monte Carlo dropout). Not tried.
- *How big is it?* 60.1 million parameters, more than the original pix2pix
  U-Net (about 54 million) because of the FiLM layers and the condition encoder.
  That is large for 4,000 training images, which is one reason for the gap
  between training and test accuracy.
- *Alternatives?* A smaller U-Net with fewer channels; attention in the
  bottleneck for long-range interactions; a vision transformer on patches; or
  a non-learned geometric prior such as shifting the height map along the sun
  direction (a shadow is the height map projected along the sun ray), which
  could be built into the first layers.

**13. What is FiLM and why use it?**

Short answer: FiLM (feature-wise linear modulation) lets the condition scale and
shift every feature channel: y = γ · IN(x) + β, where γ and β are predicted
from the sun embedding. It is applied in 13 blocks, so the sun affects both
coarse and fine scales.

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

Follow-ups:
- *Why zero initialisation and + 1?* At the start γ = 1 and β = 0, so FiLM is the
  identity and the network starts as a plain U-Net; the condition is introduced
  gradually as the linear layer learns. Starting from random γ could scale some
  channels to near zero or very large and make early GAN training unstable.
- *Why not just concatenate the sun vector to the input image as extra
  channels?* That works too (the bottleneck does something similar), but the
  information then has to survive through all layers. FiLM injects it at every
  scale. Also, concatenating constant channels to an image is wasteful: 36
  constant maps of 256×256.
- *Why InstanceNorm under FiLM?* Instance normalisation removes each image's
  own channel statistics, and FiLM then sets them according to the sun. This is
  the same idea as conditional instance normalisation and AdaIN in style
  transfer, where the "style" decides the statistics. BatchNorm would mix
  statistics across images with different suns in the same batch.
- *Does InstanceNorm throw away useful information?* It can: the mean of a
  feature map over the image (for example "how built-up is this scene") is
  removed at every normalised layer and only comes back through β, which
  depends on the sun, not the scene. The unnormalised enc1 skip and the
  bottleneck keep some of it. Worth an experiment.
- *How do you know the model actually uses the sun?* The symmetry probe: with
  the same height map, reversing the order of the vectors changes the output a
  lot (exact accuracy 59.8% → 29.0%), so the condition clearly drives the
  output. A cleaner test: same scene, morning-only vs afternoon-only sun, and
  check the shadows move to the correct side.
- *Alternatives to FiLM?* Cross-attention between image features and the 9 sun
  tokens (each pixel can attend to the hours relevant to it), or SPADE-style
  spatial modulation. Cross-attention over the 9 vectors would also remove the
  dependence on slot order.

**14. Why keep the 9 vectors in order instead of averaging them? Is that
right?**

Short answer: averaging collapses the sun path into one direction, so "sunny in
the morning, shaded in the afternoon" is lost, and very different days can
average to the same vector. Keeping them in order keeps all the information.
But the symmetry probe showed a cost: the label doesn't depend on the order, yet
the model does. The physically right structure is a set function.

Results of `training/probe_symmetry.py` (50 test scenes):

| Variant | What changes | Label | Exact | Within ±1 h |
|---|---|---|---|---|
| original | nothing | same | 59.8% | 91.1% |
| reversed_order | the 9 vectors in reverse order | same | 29.0% | 72.7% |
| mirror_east_west | image left-right, x negated, order reversed | mirrored | 50.2% | 84.7% |
| mirror_north_south | image top-bottom, y negated | mirrored | 52.5% | 86.4% |

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

Follow-ups:
- *What is a set function here?* Sum (or mean) over hours of a learned function
  of each vector, then a second network: ρ(Σ φ(sᵢ)). This is the DeepSets form.
  Unlike plain averaging of the raw vectors, the learned φ lifts each direction
  into many features before summing, so no information is lost, and the output
  is the same for any order.
- *Why does the physics say set?* Direct sun hours = Σᵢ lit(pixel, sᵢ). It is a
  sum over the hours of a per-hour quantity, so order is irrelevant by
  definition. The per-hour shadow mask idea in question 10 is the most literal
  version.
- *Why did reversing the order hurt so much?* In training slot 0 always held the
  earliest daylight hour, so the network learned slot-specific weights. Reversed
  input is outside anything it has seen. It is a clear sign of shortcut
  learning: correlations that hold in the data but not in the physics.
- *Why are the mirrored scenes 7–10 points worse than the originals, when they
  are physically valid inputs?* Mirrored scenes (for example, the sun moving
  through the north at a northern latitude) are rarer or absent in the
  training distribution, so this measures how far the model has learned
  geometry versus the specific sun paths it saw. The latitude range explains
  part of it: training covers 35°S to 65°N, so mirroring a scene at 50°N
  north-south imitates a site at 50°S, a sun path the model never saw.
- *How would you fix it cheaply?* Augment with physically valid transforms:
  rotate and mirror scenes together with their sun vectors, and shuffle the
  order. Zero labelling cost, because the labels transform exactly.

### F. Loss functions and the discriminator

**15. What exactly is the loss?**

Short answer: generator loss = least-squares GAN loss + 50 × L1. L1 keeps the
values right everywhere; the GAN term penalises outputs that a discriminator
can tell apart from real simulation results, which mainly sharpens edges.

```python
criterionGAN = nn.MSELoss()      # least-squares GAN (LSGAN); pix2pix uses cross-entropy
criterionL1 = nn.L1Loss()
L1_LAMBDA = 50.0                 # pix2pix uses 100

fake_B = netG(real_A, vec)
pred_fake_for_g = netD(real_A, fake_B, vec)
loss_G_GAN = criterionGAN(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
loss_G_L1 = criterionL1(fake_B, real_B) * L1_LAMBDA
loss_G = loss_G_GAN + loss_G_L1
```

Follow-ups:
- *Why L1 and not L2 for the reconstruction?* When the model is unsure between
  two values, L2 is minimised by their mean, L1 by their median. The mean of
  "shadow" and "no shadow" is a grey band; the median picks one, so L1 blurs
  less. L1 is also less dominated by a few large errors.
- *Why a GAN at all, if L1 alone gives a reasonable map?* L1 alone still blurs
  shadow edges where the model is uncertain. The adversarial term rewards
  outputs that look like real Ladybug maps, which have sharp edges.
- *Is a GAN the right tool for a physics surrogate?* This is the critical
  question, and the honest answer is: not obviously. A GAN rewards looking
  realistic, not being correct. It can draw a sharp edge in the wrong place,
  which looks better than a blurry edge in the right place but scores worse. For
  a surrogate, sharpness is only useful if it is accurate. I would now compare
  against L1 only, and against a classification or ordinal loss, which avoid
  the blur problem without an adversary.
- *Why LSGAN instead of the original cross-entropy loss?* With cross-entropy,
  once the discriminator is confident the generator's gradient becomes very
  small (the sigmoid saturates). The least-squares loss keeps penalising
  samples by how far they are from the target, so gradients don't vanish as
  easily and training is usually more stable. Not verified by an ablation here.
- *Why λ = 50 instead of 100?* A lower L1 weight gives the adversarial term more
  relative influence, aiming for sharper edges. Not tuned systematically; the
  honest answer is that it was a judgement call, and an ablation over
  {0 (GAN only), 10, 50, 100, L1 only} is the obvious experiment.
- *What are the magnitudes of the two terms?* The training log prints them
  (`GAN:` and `L1:`). The logs weren't kept, so say you didn't record them and
  that you would log both terms per epoch next time.

**16. What does the discriminator look like?**

Short answer: a conditional PatchGAN. It sees the height map, a sun-hours map
(real or generated) and the sun embedding, and outputs a 30×30 grid of scores,
each judging one 70×70-pixel patch.

```python
class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_nc=6, vec_dim=36, emb_dim=128, cond_ch=16):
        super().__init__()
        self.cond_enc = SunConditionEncoder(emb_dim=emb_dim)   # its own encoder
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

Follow-ups:
- *How do you get 70×70?* Work backwards from one output: the last two layers
  (kernel 4, stride 1) give 4, then 7; each stride-2 layer gives (r − 1) × 2 + 4:
  16, 34, 70.
- *Why patches instead of one score?* The global structure is already enforced by
  L1; the discriminator only needs to judge local texture and edges. A patch
  discriminator has fewer parameters and works on any image size.
- *But shadows are non-local. Can a 70×70 patch judge them?* No. It can judge
  whether an edge is sharp, not whether a shadow 150 pixels long is correct for
  this sun. That is another reason the GAN term is about appearance, not physics.
- *Why condition the discriminator on the height map and the sun?* Without them
  it could only ask "does this look like a sun-hours map", and the generator
  could output any realistic map. With them it asks "is this the right map for
  this scene and this sun". The sun goes in as 16 constant channels.
- *Why does it have its own condition encoder?* So the generator and the
  discriminator don't share parameters they optimise in opposite directions.

### G. Training

**17. What are the training settings, and why?**

```python
BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 200
SAVE_EPOCH_FREQ = 5
L1_LAMBDA = 50.0

optG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
optD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
```

Short answer: Adam with β₁ = 0.5, learning rate 1e-4 for both networks, batch
16, 200 epochs, constant learning rate. 4,000 images / 16 = 250 steps per epoch,
so the released epoch-195 model saw about 49,000 updates.

Follow-ups:
- *Why β₁ = 0.5?* β₁ controls momentum, how much of the past gradients Adam
  keeps. In a GAN the target moves because the other network keeps changing,
  so old gradients go stale; less momentum reacts faster and oscillates less.
  It is the DCGAN and pix2pix default.
- *How does this differ from pix2pix?* pix2pix uses lr 2e-4, batch 1, and keeps
  the rate constant for 100 epochs then decays it linearly to zero over another
  100. Here: lower rate, larger batch, no decay. The decay usually helps the last
  part of training; not having it is a missed detail.
- *Why batch 16 with InstanceNorm?* InstanceNorm doesn't depend on the batch, so
  batch size only affects gradient noise and speed. 16 fits in memory at 256 px.
- *How long did it take, on what?* On one desktop: Intel Core i7 (9th gen), RTX 3060 Ti (8 GB), 32 GB RAM. I didn't record the training time.
- *Why 200 epochs, and how do you know it wasn't over- or under-trained?* 200 is
  the pix2pix schedule. There was no validation set, so I can't show the best
  epoch. The training subset scores 70.3% exact against 59.3% on test, so
  there is some overfitting; with a validation split I would pick the
  checkpoint by validation within-1-hour accuracy.

**18. Walk me through one training step.**

```python
for i, batch in enumerate(dataloader):
    real_A = batch['A']; real_B = batch['B']; vec = batch['vec']

    # 1) discriminator
    optD.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_B = netG(real_A, vec)                  # no graph for G here
    pred_real = netD(real_A, real_B, vec)           # target 1
    pred_fake = netD(real_A, fake_B.detach(), vec)  # target 0
    loss_D = (criterionGAN(pred_real, torch.ones_like(pred_real)) +
              criterionGAN(pred_fake, torch.zeros_like(pred_fake))) * 0.5
    loss_D.backward()
    optD.step()

    # 2) generator
    optG.zero_grad(set_to_none=True)
    fake_B = netG(real_A, vec)                      # again, now with a graph
    pred_fake_for_g = netD(real_A, fake_B, vec)     # target 1: fool D
    loss_G = criterionGAN(pred_fake_for_g, torch.ones_like(pred_fake_for_g)) \
             + criterionL1(fake_B, real_B) * L1_LAMBDA
    loss_G.backward()
    optG.step()
```

Follow-ups:
- *Why run the generator twice?* The first pass is inside `no_grad` and only
  produces inputs for the discriminator, which saves memory. The second builds
  the graph needed for the generator update. Reusing one pass with `detach()`
  for D is the more common pattern and saves one forward pass.
- *Why multiply the discriminator loss by 0.5?* pix2pix does it to slow the
  discriminator relative to the generator.
- *Does the generator update change the discriminator?* `loss_G.backward()`
  computes gradients for D's parameters too, but only `optG.step()` is called,
  and `optD.zero_grad()` clears them at the start of the next step.
- *Which mode is the generator in during training?* `train()`: dropout is on and
  InstanceNorm uses the current image's statistics (InstanceNorm has no running
  statistics by default, so train and eval normalise the same way; only dropout
  differs).

**19. Was GAN training unstable? How did you monitor it?**

Short answer: progress was judged by the printed losses and by an HTML page that
shows input, prediction and ground truth for one fixed training sample every 5
epochs. If you don't remember specific instability, say so: you watched the
losses and the preview images and didn't keep logs, which is the weakness the
follow-ups below are about.

```python
if epoch % SAVE_EPOCH_FREQ == 0 or epoch == EPOCHS - 1:
    netG.eval()
    with torch.no_grad():
        fixed_fake = netG(fixed_A, fixed_vec)
    save_visuals(epoch, fixed_id, fixed_A, fixed_fake, fixed_B, IMG_DIR)
    update_html(WEB_DIR, IMG_DIR)
    netG.train()
```

Follow-ups:
- *What is wrong with monitoring one training image?* It is a training image, so
  it shows memorisation as progress, and one image is not a statistic. GAN loss
  values also don't tell you about quality: both losses can stay flat while
  outputs improve or degrade.
- *What should you monitor?* Within-1-hour accuracy and MAE on a held-out
  validation set every few epochs, plus a fixed panel of validation images
  covering short days, low sun and high-rise scenes.
- *What failure modes do GANs have?* Discriminator wins too early and the
  generator gets no useful gradient; oscillation; mode collapse (the same output
  for different inputs). Mode collapse is less of a problem in pix2pix-style
  training because the L1 term ties each output to its own label.

### H. Data

**20. How was the dataset generated?**

Short answer: a Grasshopper script (`gh_scripts/data-generator.gh`) randomly
subdivides a site and extrudes each plot into a building, sampling height,
density and footprint shape (polygon, rectangle, circle, L-shape, …) by
category. For each scene it picks a random location and date, computes direct
sun hours from 9:00 to 17:00 with Ladybug, and saves the height map and the
Ladybug image side by side, plus the sun vectors in a log file. 4,000 scenes for
training, 200 for testing.

```python
# each training file is one 1024×512 PNG: height map left, label right
w_half = img.shape[1] // 2
real_A, real_B = img[:, :w_half, :], img[:, w_half:, :]
vec = self.vector_map.get(img_id, torch.zeros(VEC_DIM, dtype=torch.float32))
```

What the data actually contains (measured on 2026-10-20):

| | Training (4,000) | Test (200) |
|---|---|---|
| Scenes with all 9 daylight hours | 56% | 84% |
| Scenes with 3–6 daylight hours | 13% | 4.5% |
| Median of the day's highest sun altitude | 37.6° | 54.6° |
| Site share of the image | | 19–44% (median 30%) |
| Building share of the site | | 2.5–30% (median 13%) |
| Height categories | mixed | 51 low-rise, 49 mid-rise (ids 0000–0099), 100 high-rise (0100–0199) |
| Shares site outline and sun vectors with a training scene | | 199 of 200 (current files, not the original test set) |

Follow-ups:
- *Is the test set representative of the training set?* Not quite: the test set
  has many more full summer-like days (84% vs 56% with 9 daylight hours) and a
  higher sun. Low sun means long shadows, the hardest case, so the test set is
  probably easier than the training distribution and the reported accuracy may
  be optimistic for winter.
- *Are test and training scenes independent?* Describe the set the numbers come
  from, as it is: "The test scenes reuse the site outlines and sun conditions
  of training scenes, with new building layouts. So the numbers show the model
  handles buildings it hasn't seen, not new sites, latitudes or dates. A
  cleaner test would hold out whole locations and dates, or a whole building
  category." No need to bring up the lost original test set unless asked.
- *Is the test set representative of real cities?* No. It comes from the same
  generator as the training data, so it measures interpolation within that
  family of forms, not generalisation to real, irregular cities. That is the
  biggest gap in the evaluation. The test I would add: real city blocks from
  OpenStreetMap building footprints with heights, labelled by Ladybug.
- *Why 4,000? How did you choose it?* Not chosen systematically; it was what was practical to generate (tens of seconds to a minute per scene). A learning curve
  (accuracy vs. 500, 1,000, 2,000, 4,000 training scenes) would show whether
  more data still helps.
- *Ranges of location and date?* Latitude 35°S–65°N, longitude anywhere, date anywhere in the year, uniformly random; 9:00–17:00 in local solar time.
- *How does Ladybug compute the label?* For each grid point, a ray toward each of
  the sun positions; count unblocked rays. Grid 5 m, offset 0.2 m, hourly.
- *What does a 5 m grid mean for the label?* The ground truth itself only has
  one value per 5 m cell, while one model pixel is 2.8–6.1 m. So the label is
  as coarse as the model's pixels, and a shadow edge in the label is only
  accurate to about one cell. Part of the one-hour errors at boundaries is
  therefore disagreement with a coarse label, not necessarily model error: the
  model can't be judged more precisely than its ground truth.
- *Why 5 m and not 1 m?* With 600–1300 m sites, 1 m would mean up to 1.6
  million points and 9 rays each per scene, 25 times the cost of 5 m, for
  thousands of scenes. 5 m keeps data generation affordable; the price is the
  coarse label.
- *Are roofs part of the label?* Yes: roof pixels are coloured, mostly 8–9 hours.
  They make up 13% of the site pixels and are less accurate than ground pixels
  (MAE 0.92 h vs 0.50 h). A likely reason: roof edges, where the height map and
  the label disagree by a pixel, are a large share of a small roof. Not tested.
- *Why no augmentation?* Not done. Valid augmentations here must transform the
  sun with the scene: rotate the image by 90° and rotate the vectors about z;
  mirror the image and mirror the vectors. Colour jitter or random crops would
  be wrong (crops change what can cast a shadow into the frame).
- *Could the test set leak into training?* Not by id (separate folders and
  logs), but a random generator can produce near-duplicates. With continuous
  random parameters exact duplicates are unlikely.

### I. Evaluation

**21. How did you compute the accuracy?**

Short answer: with `training/evaluate_model.py` on the 200 held-out scenes. Both
the prediction and the ground truth are snapped to the nearest of the 10 legend
colours in Lab space, which turns them into hours 0..9, and only pixels inside
the site are compared. I report exact class, within ±1 hour, and mean absolute
error in hours, against a no-shadow baseline.

```python
LEGEND_HEX = ["0000ff", "3a00c4", "770088", "b2004c", "e80014",
              "ff2b00", "ff6500", "ffa100", "ffdc00", "ffff00"]   # index = hours

def to_hours(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 1, 3).astype(np.float32)
    d = ((lab - LEGEND_LAB[None]) ** 2).sum(-1)       # squared distance to each colour
    return d.argmin(1).reshape(rgb.shape[:2])

site = ~np.all(A_s == 255, axis=2)                    # inside the site
t = to_hours(B_s)[site]
p = to_hours(pred_rgb)[site]
err = np.abs(p - t)
maxh = min(len(e["vectors"]), 9)
base = np.full_like(t, maxh)                          # no-shadow baseline
```

| Metric (3.97 M site pixels) | Model | No-shadow baseline |
|---|---|---|
| Exact hour class | 59.3% | 39.8% |
| Within ±1 hour | 90.8% | 53.6% |
| MAE | 0.56 h | 2.18 h |

Follow-ups:
- *Why Lab?* Euclidean distance in RGB doesn't match perceived colour
  difference; Lab is designed so that it roughly does. The model's colours are
  never exactly on the legend, so the match to the nearest legend colour must be
  reliable. Strictly, the best space would be the one the legend was built in:
  an interpolation in hours, which is what predicting hours directly avoids.
- *Why this baseline?* It is what you get by ignoring the buildings, so the gap
  shows how much the model learned about shadows. A stronger baseline would be
  more convincing: for example a simple geometric shadow projection, or a plain
  U-Net trained with L1 only.
- *Why is exact accuracy only 59%?* Sun hours change in steps of one hour across
  a shadow edge, and a one-pixel shift of an edge flips a pixel's class. 76% of
  site pixels are within 2 pixels of a class boundary; there exact accuracy is
  49%, against 91% inside uniform regions. Within ±1 hour and MAE are more
  meaningful for design.
- *Pixel-weighted or image-weighted?* Pixel-weighted over the whole test set;
  per-image mean exact accuracy is 58.5%, and the worst image is 37.9%, so
  there is real variation between scenes.
- *Is 0.56 h good enough for design?* It depends on the decision. For "is this
  courtyard above or below 2 hours", what matters is the error near the
  threshold. A decision-level metric would be: the share of area classified on
  the correct side of the threshold, and the error in total area above it.
- *What about statistical uncertainty?* With 200 scenes, a bootstrap over scenes
  would give a confidence interval; scene-to-scene variation dominates, not
  pixel count. Not computed.

**22. Where does the model fail?**

Short answer: at shadow edges, in high-rise scenes, on roofs, and in the low
range of hours, where it tends to predict too much sun.

| Breakdown | Exact | Within ±1 h | MAE |
|---|---|---|---|
| Near class boundaries (76% of pixels) | 49.2% | 88.2% | 0.70 h |
| Inside uniform regions (24%) | 90.8% | 99.2% | 0.10 h |
| Low- and mid-rise scenes (100) | 71.4% | 94.1% | 0.40 h |
| High-rise scenes (100) | 46.8% | 87.5% | 0.73 h |
| Ground pixels (86%) | 60.4% | 92.3% | 0.50 h |
| Roof pixels (13%) | 52.9% | 81.2% | 0.92 h |
| Training subset (49 scenes) | 70.3% | 95.4% | 0.38 h |

```python
# evaluate_model.py: class-boundary band, the true class changes within 2 px
T8 = T.astype(np.uint8)
band = (cv2.dilate(T8, kernel) != cv2.erode(T8, kernel)) & site   # kernel 5×5
for name, msk in (("near_class_boundary", band), ("uniform_region", site & ~band),
                  ("ground", ground), ("roof", roof)):
    er = np.abs(P[msk] - T[msk])
    ...
sc = scenes["high_rise" if (A_s[..., 0] < 128).any() else "low_rise"]
```

Follow-ups:
- *Why are high-rise scenes harder?* Longer shadows, more overlapping shadows
  from several buildings, and more pixels in partial shade (values in the middle
  of the range), which are the hardest classes. Possibly also clipping: towers
  above 250 m all look like 250 m.
- *What is the bias pattern?* For true values of 0 to 3 hours, 36–44% of pixels
  are predicted too high and at most 26% too low, so shaded areas come out too
  sunny.
  From 4 to 8 hours errors are balanced. The 9-hour class has the highest recall
  (82%) because it is a third of all pixels and the easiest (no shadow). Classes
  2–6 have recall around 39–47%.
- *Why would shaded areas be predicted too sunny?* Class imbalance: 33% of
  pixels are 9 hours and only 3.8% are 0 hours, so the model is pulled toward
  sunny values. A weighted loss or the per-hour mask formulation would address
  it.
- *Is the gap between training and test overfitting?* Partly: 11 points in exact
  accuracy, 4.6 in within ±1 h. The test set is also distributed differently
  from the training set (question 20), which muddies the comparison.
- *What does the symmetry probe add?* It tests something accuracy can't: whether
  the model behaves consistently with the physics on inputs that should be
  equally easy. The order sensitivity is a structural flaw, not noise.

### J. System and deployment

**23. How does the whole system run?**

Short answer: when the geometry changes, Grasshopper paints the meshes by height,
captures a 512×512 screenshot of the Top view and sends it to a FastAPI backend
(`POST /predict`); a second script sends the mesh itself to `/sync_model` so the
browser can show it in 3D. The browser computes the sun
vectors for the chosen place and date and sends them over WebSocket. The backend
runs the model, converts the colours to the display palette, computes
statistics, and pushes the image and the numbers to the Three.js page, which
maps the result onto the 3D model.

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

Follow-ups:
- *How does the mesh get to the browser?* As JSON lists of vertex coordinates
  and triangle indices; quads are split into two triangles:

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

  Plain JSON is fine for a design-sized model; a large city mesh would be
  better sent in a binary format (for example glTF) and only when it changes.
- *Why is the model loaded once at startup?* Loading 60 M parameters takes
  seconds; per request it would destroy the latency.
- *Why WebSocket and not polling?* Results are pushed to every open dashboard as
  soon as they exist, in both directions (sun changes from the browser, geometry
  from Grasshopper), without repeated requests.
- *What happens if two updates arrive while the model is running?* They are
  processed in order and each one triggers inference; there is no
  "drop stale requests" logic. For real-time editing, keeping only the latest
  request would be better.
- *Why the 1-pixel erosion of the mask?* Pixels at the site border mix the
  background and the site in the raster, so their predicted colour is
  unreliable; eroding by one pixel removes them from the statistics.
- *Why the Gaussian blur and the saturation boost?* The blur reduces
  single-pixel noise in the counts; the 1.4× saturation makes the display
  easier to read. But both happen before the dashboard statistics are computed,
  so those statistics can differ slightly from the evaluation script, which
  uses neither. Cleaner: compute the statistics from the class indices before
  any display processing.
- *How do you stop users trusting a wrong prediction?* Currently nothing: on an
  unfamiliar form the model still outputs a plausible map. Options: an
  out-of-distribution check on the input (for example distance of the height
  map's features from the training set), Monte Carlo dropout or an ensemble for
  per-pixel uncertainty, and a button to run the real simulation on demand.

### K. Limitations, research directions, reflection

**24. What is the biggest limitation?**

Short answer: generalisation. All training and test scenes come from my own
generator, so I can't claim the model works on real cities, and it can't say
when it is unsure. The symmetry probe shows it also learned shortcuts that the
physics doesn't allow.

Follow-ups:
- *Ranked list?* (1) No test on real city forms. (2) No uncertainty. (3) Order
  dependence on the sun vectors. (4) 2.5D input, no context outside the site,
  no scale information. (5) Output in colours instead of hours; GAN loss that
  rewards appearance. (6) No validation set, no ablations, test set distributed
  differently from training.
- *Which one would you fix first, and how?* The evaluation on real cities,
  because it decides whether everything else is worth doing: take a few hundred
  real blocks with building heights, label them with Ladybug, and measure.

**25. If you did it again, what would you change?**

Short answer, in order:
1. Predict hours (or 9 per-hour shadow masks) instead of colours; drop or
   justify the GAN with an ablation.
2. A set encoder for the sun vectors (or per-hour masks), and physically valid
   augmentation (rotations and mirrors with the sun).
3. A validation split with model selection; a learning curve; ablations of the
   loss terms.
4. Height map with fixed metric scale, plus a separate analysis mask, and
   context buildings outside the site.
5. A test set of real cities, and per-pixel uncertainty.

**26. How does this connect to what you want to do in a PhD?**

Adapt to the position, but the common thread: fast learned models of physical
environments that are trustworthy. This project showed me three open problems
from the inside: generalisation beyond the training generator, uncertainty
(the model is deterministic and confident everywhere), and building physical
structure into the model (the order and symmetry results). For each position
card, pick the one that matches the group: probabilistic prediction for KTH,
3D representations for MPI and DFKI, agents and generated environments for
Chalmers, model choice and efficiency for Lund and TU Delft TPM.

**27. What was the hardest part?**

Use your own experience if you remember it. Answers that are true from the
code and the data: getting training and inference to see exactly the same input (the
masking and slot-order bugs, NEAREST resizing, the colour palette); and
evaluating honestly once a pretty demo existed, which changed the headline
number from an estimate of 80% to a measured 59% exact / 91% within one hour.

**28. What did you learn?**

That a model that looks right is not a model that is right: the outputs looked
convincing long before they were measured. Evaluation design (baselines,
breakdowns, symmetry checks) told me more about the model than the training
curves. And that physical structure (what the label depends on and what it
doesn't) should guide the architecture.

**29. What is novel compared with pix2pix?**

Honest answer: the method is not new. What differs from plain pix2pix is the
problem formulation and the conditioning: a physical condition (sun vectors
over a day) injected through FiLM at 13 scales and into the discriminator, so
one model covers any location and date instead of one model per sun position.
The evaluation and the analysis of where it breaks are the part I would build
research on.

**30. Rapid-fire checks (one-sentence answers)**

- Input size? 256×256×3, normalised to [−1, 1].
- Condition size? 36 = 9 × 3 + 9.
- Embedding size? 128.
- Generator parameters? 60.1 million.
- Encoder depth? 8 down-sampling steps, 256 → 1.
- FiLM blocks? 13 (enc2–7, dec1–7).
- Normalisation? InstanceNorm, no affine; FiLM provides scale and shift.
- Dropout? 0.5 in dec1–dec3, off at inference.
- Discriminator output? 30×30 patch scores, 70×70 receptive field.
- Loss? LSGAN + 50 × L1. D loss × 0.5.
- Optimiser? Adam, lr 1e-4, β = (0.5, 0.999), batch 16, 200 epochs, no decay.
- Released checkpoint? Epoch 195 (0-indexed), about 49,000 updates.
- Data? 4,000 training, 200 test, from a Grasshopper generator, labels from
  Ladybug.
- Result? 59.3% exact, 90.8% within ±1 h, MAE 0.56 h; baseline 39.8%, 53.6%,
  2.18 h.

**31. Questions you should be ready for that test understanding, not memory**

Answer these out loud; the answers follow from the material above.
- If you double every building's height, what should happen to the output, and
  does your input encoding let the network know?
- If you move the site from 40°N to 40°S on the same calendar date, what changes
  in the sun vectors and in the label?
- Why is the 1×1 bottleneck not enough on its own, and why are skip connections
  not enough on their own?
- What would happen if you trained with L1 only? With the GAN only?
- How would you get a confidence value for each pixel without retraining?
  (Monte Carlo dropout.) With retraining? (Ensembles; classification output;
  a probabilistic model such as a diffusion model.)
- How would you check that the model is not just memorising the generator's
  building shapes? (Real-city test set; a held-out building category.)
- A reviewer says "this is just pix2pix". Respond.

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
