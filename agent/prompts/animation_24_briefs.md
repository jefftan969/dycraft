# Animation briefs — the 24 benchmark briefs

The task text of every brief in the benchmark, in one file, collected and
grouped; nothing else is added to them.

**What a brief is.** One paragraph of agent-facing prose naming the initiator
and the goal, the scene's initial state and the props the activity needs, the
activity in causal order, and the terminal state it settles into. Every one is
long-horizon: on the order of a dozen or more beats in causal order, with a
state that accumulates across them. Fifteen of the 24 carry a complication the
initiator has to handle — a shelf that rocks, a parcel that skews, a name
called twice, a two-pace stay the dog breaks, a label that feeds short — and
where one is present the recovery is usually the point of the brief rather
than an incident in it.

**What is deliberately absent.** Duration, frame rate, aspect ratio, camera
path, visual style and production workflow: those belong to the harness that
ingests a brief, not to the task text. So does the wall-clock budget — the
budget paragraph a run carries is appended at launch and is not part of any
brief here.

**Grouped by what the arm has to build, in three sections:**

| | briefs | what the arm is given, and what it must author |
|---|---|---|
| G1 humanoid | B01–B08 | the character is the Unitree G1, taken from the asset library; the scene is authored |
| Non-G1 character | B09–B16 | people and animals authored and rigged from nothing |
| Physical mechanism | B17–B24 | no characters at all; the mechanism is the performer |

**The G1 is retrieved, not handed over.** The briefs name the robot ("a
Unitree G1 humanoid robot with Dex1 grippers") and the agent finds it in the
asset library (`references/assets/INDEX.md` in the run; bundle
`g1_with_dex1_v3`) and places it with `load_asset`.

Duration, frame rate, camera path, visual style and the wall-clock budget are
absent for the same reason: they belong to the harness that ingests a brief.

**Sizing.** Each set is sized to a runtime. The short set delivers within 6–8
hours and the medium set within 12–24. The long set is the paper's and stays
as written; the benchmark gives it a 24-hour budget by default. Each
section's table gives every brief its plan, predicted film and expected
runtime:
- **plan:** every state the brief implies, timed at the `humanoid_motion`
  skill's rates (`python -m dycraft.benchmark.sizing`; the beats are in
  `benchmark/sizing_beats.py`);
- **predicted film:** the plan × 1.28, because finished films run slower than
  the bare sum;
- **expected runtime:** 2.9 h + 3.9 h per predicted minute, with the 10th–90th
  percentile in brackets. It is fitted on recorded runs that delivered one
  film covering their brief: GPT-6 at `xhigh`, characters from the asset
  library, the G1 or none. Past about 1.5 minutes of predicted film it is an
  extrapolation.

---

## G1 humanoid — B01–B08

The character is handed to the arm: 38 parts, 37 joints, 0.469 x 0.375 x
1.323 m, with the anatomical overlay from
`catalogs/anatomical/g1_humanoid_dex1.json`. Each brief names the asset in its
own text, so the arm is told what it is being given in the same words that
describe the job.

| | plan | predicted film | expected runtime |
|---|---|---|---|
| B01 | 2:50 | 3:37 | 17 h (12-24) |
| B02 | 4:28 | 5:43 | 25 h (18-35) |
| B03 | 3:46 | 4:49 | 22 h (15-30) |
| B04 | 3:00 | 3:51 | 18 h (13-25) |
| B05 | 1:00 | 1:17 | 8 h (6-11) |
| B06 | 1:33 | 1:59 | 11 h (7-15) |
| B07 | 1:02 | 1:20 | 8 h (6-11) |
| B08 | 1:29 | 1:54 | 10 h (7-14) |

### B01. A humanoid cooks scrambled eggs, fridge to plate

A Unitree G1 humanoid robot cooks scrambled eggs by itself in a compact kitchen. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The preparation counter starts clear; three eggs and a portion of butter sit in the refrigerator, with a bowl, whisk, spatula, plate, and cloth on a nearby shelf and a frying pan on the stovetop. An open waste bin stands beside the counter. The robot opens the refrigerator, carries the eggs and butter to the counter, retrieves the bowl, and cracks each egg over it, separating the shells and setting the empty halves aside. It steadies the bowl with one hand and whisks with the other, then centers the pan on the burner, turns on the heat, and adds the butter. Once the butter melts, it lifts and tilts the bowl to pour the eggs into the pan, sets the emptied bowl down, and scrapes and folds the mixture with the spatula as it thickens into soft curds. It turns off the burner, brings over the plate, and tilts the pan while guiding the eggs onto it, then returns the pan to the stove. Finally, it drops the shells into the bin, wipes the preparation counter, puts down the cloth, and closes the refrigerator.

### B02. Flat-pack office chair assembly

A Unitree G1 humanoid robot assembles a flat-pack office chair by itself in an open-plan office bay. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. A sealed carton stands on the floor; the robot cuts its tape with a safety cutter, folds the four top flaps open, and lays out a five-arm star base, five casters, a gas-lift column with its plastic sleeve, the moulded seat, the backrest and its mounting spine, bolts and washers, bolt cover caps, and an Allen key. It presses each caster into a socket of the upturned base, rights the base, slides the sleeve over the column, and inserts the column into the base hub. It places the seat upside down on a low workbench, bolts the backrest spine to its underside, tightens the bolts, and fits the cover caps. It turns the upper assembly upright, lifts it over the column, and lowers it onto the taper. The robot sits briefly to seat the connection, tests the swivel, height lever, and backrest tilt, then stands, puts away the tools, flattens the empty carton beside the recycling bin, and rolls the finished chair to a desk.

### B03. Loading a pottery kiln

A Unitree G1 humanoid robot loads mixed pottery into a front-loading kiln in a pottery studio. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The kiln is cold and its door is open. A nearby trolley holds bowls, cups, a tall vase, plates, kiln shelves, support posts of several heights, a clearance gauge, and a cone pack. The robot places three posts inside the kiln and lowers the first shelf onto them. It presses lightly at the shelf edges, notices a wobble, lifts the shelf, replaces the uneven post, and checks it again. It loads four bowls with gaps between them, checks the gaps with the gauge, and nests small unglazed cups inside a larger unglazed bowl. It adds taller posts and a second shelf, checks the vase against the available headroom, and places it upright. It loads plates into a rack, adds a third shelf for the remaining small pieces, and places the cone pack where it will be visible through the viewing port. Finally, it checks the clearances around the loaded shelves, withdraws its hands, closes and latches the kiln door, and returns the unused posts and gauge to the trolley.

### B04. Mopping a room toward the exit

A Unitree G1 humanoid robot mops a tiled office break room, working backward from the far corner toward the doorway. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. Chairs are tucked beneath the tables, leaving an open floor with several dusty footprints and a small coffee stain. A mop, a wheeled bucket with a wringer, and a folding wet-floor sign stand in the dry hallway. The robot places the sign beside the entrance, rolls the bucket into the room, dips and wrings the mop, and begins in the far corner. It sweeps the mop in overlapping figure-eight strokes, then steps backward onto the remaining dry floor. After several passes, it rinses and wrings the mop and pulls the bucket closer along the dry strip. It works around the table legs, scrubs the coffee stain with shorter strokes, and continues toward the exit as the cleaned wet area expands. Near the doorway, it moves the bucket into the hallway, mops the final strip, lifts the mop clear, and steps out. It secures the mop in the bucket and leaves the warning sign beside the freshly cleaned room.

### B05. Carrying a wide panel through a doorway

A Unitree G1 humanoid robot carries a wide rectangular panel from a workshop into an adjoining room through a narrow doorway. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. A lightweight panel rests horizontally on two waist-high trestles. It is too wide to pass through the doorway broadside but can fit when turned edge-on. A second pair of trestles stands in the destination room, and the door is secured open. The robot approaches the panel, grips its two accessible edges, lifts it clear of the supports, and rotates it into an upright carrying position. It adjusts its stance and walks toward the doorway with short steps, stopping before the frame to turn the panel edge-on. It guides the leading edge through, sidesteps across the threshold while turning its torso, and brings the trailing edge clear of the frame. Once inside, it turns toward the empty trestles, pauses to align the panel, rotates it back to horizontal, and lowers it onto both supports. It releases its grips, steps back, and checks the panel’s placement before returning to a relaxed standing posture.

### B06. Sorting mail with exceptions

A Unitree G1 humanoid robot sorts a mixed bundle of mail at a mailroom sorting station. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. A counter holds a stack of envelopes addressed to six clearly labelled departments, an exception tray, rubber bands, and one small parcel. A bank of matching pigeonholes stands within reach, with a parcel shelf below. The robot squares the bundle, picks up the first envelope, turns its address toward its head, and places it into the corresponding pigeonhole. It continues through several envelopes at a steady pace until it finds one addressed to another building, pauses to inspect it, and sets it in the exception tray. It resumes sorting, then reaches the parcel and checks it against the intended pigeonhole. Finding it too large, it withdraws the parcel and places it on the lower shelf. It works through the remaining envelopes, sets aside a second misdirected item, and checks the now-empty input area. Finally, it gathers the mail from three occupied pigeonholes into separate bundles, secures each with a rubber band, and places them in a collection tray, leaving the exceptions separate.

### B07. Handling an urgent job and resuming a tray transfer

A Unitree G1 humanoid robot transfers parts between two indexed trays at a workshop bench, interrupts its work for an urgent delivery, and then resumes where it stopped. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The trays each have twelve numbered pockets arranged in two rows. At the start, parts one through six have been transferred to the destination tray, and the robot is lifting part seven from the source tray. A padded temporary holder sits between the trays, and a separate urgent-job station stands a few steps away. An amber lamp illuminates at that station while the robot holds part seven. It pauses, lowers the part into the temporary holder, and places a small marker beside destination pocket seven. It turns, walks to the urgent station, picks up a red component, carries it to a dispatch cradle, and presses the acknowledgement button. When the lamp goes out, it returns to the original bench, checks the marker, retrieves the parked part, and places it into pocket seven. It removes the marker, transfers parts eight through twelve in order, checks the empty source tray and completed destination tray, and steps back from the bench.

### B08. Carrying a loaded tray through a crowded room

A Unitree G1 humanoid robot carries a tray of drinks from a service counter to a collection table across a busy office lounge. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. Six partially filled glasses stand beside a rigid serving tray on the counter. Several people sit at tables along the route, leaving an aisle with one narrow gap between chairs. The robot places the glasses onto the tray, lifts it with both hands, and turns toward the collection table. It walks between the occupied tables until a seated person pushes a chair backward into its path. The robot stops, lets the glasses settle, and waits while the person notices it and pulls the chair inward. It resumes, turns its body sideways to pass through the narrow gap, and brings the tray through at a steady height. Beyond the gap, a passing person lightly bumps its upper arm. The robot pauses, bends its knees slightly, adjusts its stance, and steadies the tray before continuing. At the collection table, it lowers the tray onto the surface, releases its grip, and arranges the six glasses one by one on coasters. It then picks up the empty tray and returns toward the service counter.

---

## Non-G1 character — B09–B16

No character asset exists for any of these. People, a dog, a crow and a
macaque are authored and rigged by the arm itself, which makes them
substantially harder than the eight above and is worth holding in mind when
results are compared across the 24.
The runtimes below are for characters taken from the asset library. Authored
from nothing, they take longer and are less reliable.

| | plan | predicted film | expected runtime |
|---|---|---|---|
| B09 | 3:23 | 4:20 | 20 h (14-28) |
| B10 | 1:51 | 2:22 | 12 h (8-17) |
| B11 | 3:04 | 3:55 | 18 h (13-25) |
| B12 | 3:02 | 3:52 | 18 h (13-25) |
| B13 | 0:46 | 0:59 | 7 h (5-9) |
| B14 | 0:44 | 0:57 | 7 h (5-9) |
| B15 | 1:12 | 1:32 | 9 h (6-12) |
| B16 | 0:54 | 1:09 | 7 h (5-10) |

### B09. Moving an upright piano up a stairwell

Three movers carry an upright piano up two flights of stairs in an apartment building, negotiating a tight turn at the half-landing. The piano stands in the entrance hall beside a stair-climbing dolly, moving blankets, and straps. The movers wrap the case, secure it to the dolly, and tip it backward together, with one mover above guiding and two below supporting the load. They find their balance, approach the first step, and work up the flight one step at a time, pausing midway to adjust their grips. At the half-landing, they begin turning toward the second flight, but the piano’s case meets the space occupied by the newel post. They stop, back off slightly, and bring the piano almost upright to make room for the turn. The upper mover changes position while the other two hold the load, then they pivot the piano and reposition the dolly toward the next flight. They resume climbing, reach the upper landing, and lower the piano upright onto the floor. Finally, they release the straps, withdraw the dolly, remove the blankets, and move the piano into its position against the hallway wall.

### B10. A coffee order from arrival to handoff

A barista serves a customer from ordering to collecting a takeaway coffee in a small café. The counter contains a till, paper cups, a marker, an espresso machine, a grinder, a milk pitcher, lids, sleeves, and a cleaning cloth. The customer approaches the till, orders a milk coffee, and gives their name. The barista marks a cup, repeats the order, and takes payment while the customer moves to the collection end of the counter and looks at their phone. The barista grinds and tamps the coffee, locks the portafilter into the machine, and starts the shot. While it extracts, they pour milk into the pitcher and steam it, then remove the cup and pour in the milk. They press on a lid, fit a sleeve, and call the customer’s name. The customer does not respond. The barista glances toward them and calls again, raising the cup slightly. The customer looks up, puts away their phone, and approaches. The barista places the drink within reach, the customer takes it and leaves, and the barista wipes the steam wand and counter before returning the pitcher to its place.

### B11. A film crew prepares and executes a shot

A small film crew sets up a dolly shot in a furnished studio interior. Equipment cases, three track sections, a camera dolly, a lamp, a flag, stands, sandbags, wedges, and marking tape are arranged along the edge of the set. A grip opens a stand, mounts the flag, and secures the base with a sandbag while a gaffer positions the lamp and adjusts its direction. At the same time, a camera assistant lays the track sections end to end, joins them, and levels the track with wedges. The crew places the dolly on the rails and mounts the camera. The gaffer takes a light reading beside the actor’s position and adjusts the lamp while the focus puller measures two distances and tapes marks beside the track. An actor steps onto their floor mark for a rehearsal. The dolly grip pushes the camera through the planned move, the focus puller follows the distance marks, and the crew stops to correct a bump at one track joint. They repeat the move, return to the starting positions, and make a final lighting check. The director calls for quiet, the camera starts, an assistant presents and snaps the slate, and the actor performs as the dolly travels along the track.

### B12. An unsuccessful lift followed by putting the equipment away

A weightlifter attempts a heavy deadlift in a gym, fails to complete it, and puts the equipment away. An unloaded barbell lies on a lifting platform beside four weight plates, collars, a lifting belt, a chalk bowl, and a plate rack. The lifter loads two plates on each end, secures the collars, tightens their belt, and chalks their hands. They approach the bar, set their feet, bend down, and establish their grip. On the first attempt, they pull the bar to knee height, stall, and lower it back to the platform. They release their grip, stand to take a breath, then reset their feet and try again. The second pull rises slightly higher but still stops before lockout, and they lower it once more. After a longer pause, they change their grip and make a third attempt, this time lifting the bar only a short distance before returning it to the floor. They let go, step back, and loosen their belt. Finally, they remove the collars, unload the plates one at a time onto the rack, wipe the bar, and leave it empty on the platform.

### B13. A dog fetches the newspaper

A dog retrieves a rolled newspaper from the front garden and delivers it to its owner on the porch. The owner sits beside the front door, and a newspaper secured with a paper band lies near the garden gate at the end of a short path. The dog rests beside the chair until the owner gives a verbal cue and points toward the gate. It raises its head, stands, walks to the porch steps, and trots down onto the path. Near the gate, it slows, lowers its nose, and locates the newspaper beside a plant pot. It grips the middle of the roll in its mouth, lifts it, turns, and trots back toward the house. At the steps, it slows and climbs while carrying the newspaper. On reaching the porch, it pauses to adjust its grip, then approaches the owner and raises its muzzle toward the waiting hand. The owner takes hold of the newspaper before the dog releases it. The dog sits beside the chair, the owner places the newspaper on their lap and strokes its head, and the dog settles at their feet.

### B14. A crow opens a puzzle feeder

A crow works out how to open a puzzle feeder on a low outdoor table. Food is visible behind a transparent sliding hatch, but a removable peg blocks the hatch from opening. A perch stands beside the table. The crow lands near the feeder, walks around it, and tilts its head to inspect the food through the clear panel. It pecks the panel several times, then catches the edge of the hatch with its bill and pushes it sideways. The hatch moves a little and stops against the peg. The crow steps back, looks along the hatch edge, and pecks at the protruding peg. It grips the peg with its bill and pulls, but loses its grip. It changes position, braces beside the feeder, hooks one foot around the peg, and draws it out of the retaining hole. The peg drops onto the table. The crow returns its bill to the hatch edge and slides the hatch far enough to expose the food. It reaches inside, takes a piece, withdraws its head, and hops onto the nearby perch to eat, leaving the hatch open and the peg on the table.

### B15. Teaching a dog to sit and stay

A trainer teaches a young dog to sit and stay in a quiet fenced garden. The trainer wears a treat pouch, and the dog begins standing in front of them on a clear patch of grass. The trainer holds a treat above the dog’s nose and moves it backward until the dog sits, marks the response with a short word, and gives the treat. They repeat the sit with a raised-hand signal, reward it, and then introduce an open-palm stay signal. The trainer steps backward one pace, pauses, returns to the dog, and rewards it while it remains seated. On the next attempt, the trainer takes two steps back, but the dog stands and follows. The trainer stops, guides the dog back to its original place, and asks it to sit again. They repeat the easier one-step stay and reward it, then try two steps again. This time the dog holds its position until the trainer returns. Finally, the trainer backs away three paces and briefly turns sideways before facing the dog again. The dog remains seated. The trainer returns, gives a release cue, and rewards the dog as it gets up and approaches.

### B16. A young macaque learns to crack a nut

A young macaque attempts to crack a hard-shelled nut on a flat stone in a forest clearing. Several nuts lie beside the stone, with a small cobble nearby and a larger, heavier stone a short distance away. The macaque picks up a nut, places it on the flat surface, and lifts the small cobble in one hand. It strikes the nut, but the nut rolls off the stone without opening. The macaque follows it, retrieves it, and places it back near the center. A second strike sends the nut skidding to the edge, still intact. It retrieves the nut again, sets down the small cobble, and moves to the heavier stone. It lifts that stone with both hands, brings it over the nut, and strikes, but misses to one side. The macaque lowers the stone, repositions the nut, widens its stance, and adjusts its grip. It raises the stone again and strikes squarely, cracking the shell. It sets the stone down beside the anvil, separates the shell pieces with its fingers, picks out the kernel, and sits beside the scattered fragments to eat.

---

## Physical mechanism — B17–B24

No people, animals or robot characters appear. Initiative comes from a wound
spring, a released gate, a motor, a float or a descending weight, and the
deliverable is a causal chain that reads: every stage's output is
unmistakably the next stage's input.

| | plan | predicted film | expected runtime |
|---|---|---|---|
| B17 | 0:29 | 0:37 | 5 h (4-7) |
| B18 | 0:28 | 0:36 | 5 h (4-7) |
| B19 | 0:42 | 0:54 | 6 h (4-9) |
| B20 | 0:52 | 1:07 | 7 h (5-10) |
| B21 | 0:47 | 1:00 | 7 h (5-10) |
| B22 | 0:46 | 1:00 | 7 h (5-9) |
| B23 | 0:34 | 0:43 | 6 h (4-8) |
| B24 | 0:32 | 0:41 | 6 h (4-8) |

### B17. A marble route with a counterweight lift

A mechanical marble course runs on a freestanding wooden frame with ramps, a spiral funnel, a seesaw, a counterweight lift, and a finishing cup. No people, animals, or robot characters appear. At the start, a steel marble rests behind a closed gate at the top, a second marble sits in a pocket on the raised end of the seesaw, and the lift platform waits at the bottom of its guide rails. A clockwork cam opens the starting gate, releasing the first marble down a curved ramp and into the funnel. It circles inward, drops through the center, and lands on the lowered end of the seesaw, tipping it and releasing the second marble onto a separate track. The first marble rolls off the seesaw onto the lift platform and depresses a latch. A suspended counterweight descends, raising the platform until its edge meets an upper exit ramp. The marble rolls out, crosses a short jump onto a catch ramp, and enters the finishing cup. Meanwhile, the second marble follows a longer zigzag route and arrives in the same cup a moment later. The cup tips under their combined weight, drops both marbles into a collection tray, and returns against its stop.

### B18. A branching domino circuit with a mechanical finish

A branching domino circuit unfolds across a large tabletop, combining falling tiles with levers, sliding gates, and a weighted final platform. No people, animals, or robot characters appear. A wound clockwork striker releases and knocks over the first domino, starting a line that divides into two routes. The left branch climbs a series of shallow steps and ends by pushing a lever that withdraws a locking pin from a sliding gate. The right branch follows a longer curved path around a tower of wooden blocks and ends by tipping a small cup containing a steel ball. The ball rolls toward the gate, reaches it after the pin has been withdrawn, and pushes through onto a descending track. At the bottom, it strikes a paddle connected to a second domino line on a lower level. This line crosses a narrow bridge and knocks a hinged support away from beneath a weighted platform. The platform lowers slowly through a damped linkage, pulling a cord that raises a small flag. The final dominoes settle, the ball comes to rest in a pocket, and the raised flag remains visible above the completed circuit.

### B19. A gravity-fed parcel sorting machine

A gravity-fed sorting machine processes six rigid parcels from a sloped loading rack into three collection lanes. No people, animals, or robot characters appear. The parcels have equal width but three different heights, with two parcels of each height mixed in the queue. A motor-driven escapement releases one parcel at a time onto a roller track. Each parcel passes beneath a height-sensing lever that positions a downstream diverter through a mechanical linkage, routing short, medium, and tall parcels into separate chutes. After the first two parcels have been sorted, the third turns slightly and catches between the guide rails. The drive continues through a slipping clutch until a cam-operated side paddle pushes the parcel square with the track. It moves forward, passes the height lever, and enters its collection lane. The remaining parcels follow through the same release and sorting cycle. Each lane’s end stop holds its two parcels in a short queue. When the loading rack becomes empty, a follower drops into the vacant feed position and disengages the drive clutch, leaving the escapement stopped and all six parcels sorted.

### B20. A water tank fills, tips, and resets

A tabletop water mechanism transfers water from an elevated reservoir through a tipping bucket into a lower collection tank. No people, animals, or robot characters appear. The reservoir begins full, the bucket rests upright on a pivot, and the collection tank is empty. A spring-driven timer opens the reservoir valve, sending a narrow stream into the bucket. As the water rises, the bucket’s changing center of mass carries it past its tipping point, and it rotates forward to discharge into a broad funnel. A small counterweight returns the empty bucket upright, where it begins filling again. Water from the funnel turns a paddle wheel before draining into the lower tank. The bucket completes several fill-and-tip cycles, and the wheel starts with each discharge and slows between them. A float in the collection tank rises as water accumulates. Near the final level, the float pulls a linkage that closes the reservoir valve. The remaining water in the feed channel drains into the bucket, leaving it partly filled below its tipping threshold. The paddle wheel coasts to a stop, and the system settles with the upper valve closed and the lower tank full.

### B21. An automatic bottle filling and capping line

An automatic production line fills, caps, and labels a batch of six transparent bottles. No people, animals, or robot characters appear. Empty bottles queue on an infeed conveyor beside a liquid reservoir, a cap chute, and a label reel. A rotating star wheel separates the bottles and brings the first beneath a filling nozzle. The bottle stops, the nozzle lowers, and a measured portion of colored liquid enters before the nozzle rises and the star wheel advances. At the next station, a cap drops from the chute into a holder and a rotating head presses and tightens it onto the bottle. The bottle then passes between powered rollers that turn it against a dispensed label, wrapping the label around its body. As more bottles enter, filling, capping, and labelling occur simultaneously at different stations. One label feeds slightly short, causing the label station to pause while a tension arm takes up slack and advances the reel to the next registration mark. Processing resumes, and the last bottles pass through. All six collect behind an outfeed stop, the empty infeed sensor halts the indexing wheel, and the remaining stations finish their cycles before stopping.

### B22. A drawbridge opens for a passing boat

An automated canal drawbridge opens for an unoccupied motorboat and closes again after it passes. No people, animals, or robot characters appear. The bridge begins horizontal and locked across the canal, with raised road barriers and the boat approaching slowly along the water. As the boat reaches an approach marker, warning lamps flash and the road barriers lower. Once both barriers reach their stops, locking bolts retract from the bridge deck. A geared drive turns the lifting shafts, raising the two bridge leaves while their counterweights descend into side housings. The boat slows near the opening and waits until the leaves reach their fully raised positions and the canal signal changes. It then travels between the piers, leaving a small wake that rocks two moored buoys. After its stern clears the far marker, the drive reverses and lowers both leaves toward the center. They settle onto their supports, the locking bolts slide into place, and the road barriers rise. The warning lamps go dark as the boat continues down the canal and the disturbed water gradually settles.

### B23. A spring-powered cart completes an obstacle route

A spring-powered cart travels through a mechanical obstacle course on a tabletop track. No people, animals, or robot characters appear. The cart starts behind a retaining latch with its drive spring wound, carrying a steel ball in a shallow cradle. A timer releases the latch, and the cart rolls forward, climbs a gentle ramp, and passes over a hinged platform that tilts under its weight. As it descends the far side, a fixed guide engages a lever on the cart and opens the cradle. The ball drops into a chute and rolls ahead to a locked crossing gate. It lands in a weighted cup, lowering the cup and pulling the gate upward by a cord. The cart reaches the crossing, passes beneath the raised gate, and enters a curved section that turns it back toward the start. It crosses a rough strip that slows its wheels, then climbs a final short incline using the remaining spring energy. At the top, it rolls into a padded docking stop and engages a catch. The drive spring unwinds the last fraction through a slipping clutch, leaving the cart parked and the ball held in the lowered gate cup.

### B24. A weight-driven clock runs and strikes

A weight-driven clock with exposed gears advances to the hour, strikes four times, and returns to normal running. No people, animals, or robot characters appear. The clock begins a short interval before four o’clock, with its pendulum swinging, its driving weights raised, and the hour-striking mechanism held by a latch. The escapement releases the gear train tooth by tooth as the minute hand approaches twelve. A cam on the motion works gradually lifts the striking release lever. At the hour, the lever clears its stop and releases a separate striking train, whose weight begins descending as a fly governor spins up. A pinned wheel lifts a hammer and lets it fall against a bell four times, with the hammer withdrawing after each impact. A counting mechanism advances with each strike and, after the fourth, allows the locking lever to drop into its stop. The striking train halts and the governor settles, while the main escapement and pendulum continue running. End with the hands showing just after four, the hammer at rest, and the timekeeping gears still advancing.

---

## Short versions — B01–B12

Twelve of the briefs above plan well past a 60-75 s film when every state
they imply is timed under the `humanoid_motion` skill's rules. These are the
same twelve cut to fit, plus B15S, because B15 predicts about 1:32 of film.
The short set is these thirteen plus B13, B14, B16 and B17-B24, with B13, B14
and B16 casting their animals from the asset library
(`animation_briefs_imported_characters.md`).

The set delivers within 6-8 hours. Each cut plans at about 30-40 s, under
a minute of predicted film, and every character is the G1 or comes from the
asset library: authored people and animals are what made runs of this length
unreliable.

| | plan | predicted film | expected runtime |
|---|---|---|---|
| B01S | 0:32 | 0:42 | 6 h (4-8) |
| B02S | 0:37 | 0:47 | 6 h (4-8) |
| B03S | 0:40 | 0:51 | 6 h (4-9) |
| B04S | 0:38 | 0:49 | 6 h (4-8) |
| B05S | 0:36 | 0:46 | 6 h (4-8) |
| B06S | 0:37 | 0:47 | 6 h (4-8) |
| B07S | 0:40 | 0:51 | 6 h (4-9) |
| B08S | 0:36 | 0:47 | 6 h (4-8) |
| B09S | 0:32 | 0:42 | 6 h (4-8) |
| B10S | 0:38 | 0:49 | 6 h (4-9) |
| B11S | 0:37 | 0:48 | 6 h (4-8) |
| B12S | 0:35 | 0:45 | 6 h (4-8) |
| B13 (library) | 0:46 | 0:59 | 7 h (5-9) |
| B14 (library) | 0:44 | 0:57 | 7 h (5-9) |
| B15S | 0:39 | 0:50 | 6 h (4-9) |
| B16 (library) | 0:54 | 1:09 | 7 h (5-10) |
| B17-B24 | 0:28-0:52 | 0:36-1:07 | 5-7 h (4-10) |

Each keeps what its brief exists to test and its complication, and drops the
setup and teardown around them. A version that opens mid-activity states its
entry condition in its opening sentences, and anything the cut needs of the
place — a smaller room, a nearer station, items staged within reach — is
stated there too, because a run sees only the brief. Counts the full brief
left open are fixed. B01S and B10S compress process time explicitly — the
butter is already melted, the eggs set on contact, the milk is already steamed
and the shot finishes almost at once — while every body motion keeps its own
pace. B11S is the only cut that is not a contiguous stretch of its brief: it
drops the rehearsal and the track-joint fix and goes straight to the take.

These and the medium versions below share one premise audit, which corrected
what the full briefs got wrong: seated chair tests the G1 can do with its feet
off the floor (lever and tilt seated, swivel by hand), a wobble caused by a
chipped post (three posts cannot rock a rigid shelf), a mop bucket kept
doorward of the wet floor with the last strip mopped from the hallway, a
padded mailer rather than a boxed parcel inside a squared stack, the steam
wand wiped as soon as the milk is steamed, and — in B01M — the refrigerator
closed as soon as the eggs are out. Every G1 brief takes the robot from the
asset library.

### B01S. Eggs cracked into the pan, cooked and plated

A Unitree G1 humanoid robot cooks scrambled eggs at the stove of a compact kitchen. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The robot stands at the counter beside the stove. Two eggs, a spatula and a plate are set out within reach, and a frying pan with butter already melted in it is hot on a lit burner. The robot cracks each egg over the pan, letting it fall in, and sets the empty shell halves aside. The eggs begin to set as soon as they meet the pan, and the robot scrapes and folds them with the spatula into soft curds. It turns off the burner, tilts the pan while guiding the eggs onto the plate, and returns the pan to the stove.

### B02S. A chair's column and seat fitted and sat on

A Unitree G1 humanoid robot finishes assembling a flat-pack office chair in an open-plan office bay. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The chair is partly built: the five-arm star base stands upright on its casters on the carpet, the gas-lift column with its plastic sleeve already fitted lies on the floor beside it, and the upper assembly — the moulded seat with the backrest already bolted to it — rests upright on a low workbench next to the base. The robot picks up the column and inserts it into the base hub. It lifts the upper assembly off the workbench, carries it over the column, and lowers it onto the taper. The robot sits briefly to seat the connection, pulls the height lever so the seat sinks under its weight, and stands.

### B03S. A chipped kiln post found and replaced

A Unitree G1 humanoid robot lays the first shelf in a front-loading kiln in a pottery studio. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The kiln is cold, its door is open, and three support posts already stand on the chamber floor. A nearby trolley holds kiln shelves and spare posts of several heights. The robot lowers the first shelf onto the three posts. It presses lightly at the shelf edges and feels a wobble: one post is chipped at its base and rocks under the shelf. It lifts the shelf, replaces the chipped post with a sound one from the trolley, lays the shelf back, and checks it again.

### B04S. A small kitchenette mopped backward to the door

A Unitree G1 humanoid robot mops the tiled floor of a small office kitchenette, working backward from the far corner toward the doorway. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The kitchenette is about two metres square, with a counter along one wall and a single doorway to a dry hallway, and its floor has several dusty footprints and a small coffee stain. A folding wet-floor sign already stands in the hallway beside the doorway. The robot starts in the far corner holding a freshly wrung mop, with a wheeled bucket and wringer standing on the dry floor just behind it, toward the doorway. It sweeps the mop in overlapping figure-eight strokes, then steps backward onto the remaining dry floor. It scrubs the coffee stain with shorter strokes and continues toward the exit as the cleaned wet area expands. Near the doorway, it moves the bucket into the hallway, steps back over the threshold, and mops the final strip from the hallway. It lifts the mop clear and secures it in the bucket beside the warning sign.

### B05S. A wide panel turned edge-on through a doorway

A Unitree G1 humanoid robot carries a wide rectangular panel from a workshop into an adjoining room through a narrow doorway. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. A lightweight panel rests horizontally on two waist-high trestles beside the doorway, and the robot stands at the trestles facing it. The panel is too wide to pass through the doorway broadside but can fit when turned edge-on. A second pair of trestles stands just inside the destination room, and the door is secured open. The robot grips the panel with both hands, lifts it clear of the supports, and rotates it into an upright carrying position. It adjusts its stance, steps to the doorway, and turns the panel edge-on. It guides the leading edge through, sidesteps across the threshold while turning its torso, and brings the trailing edge clear of the frame. Once inside, it turns to the empty trestles, rotates the panel back to horizontal, lowers it onto both supports, and releases its grips.

### B06S. Five envelopes sorted around a padded mailer and an exception

A Unitree G1 humanoid robot sorts a bundle of mail at a mailroom sorting station. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. A counter holds a squared stack of five envelopes addressed to clearly labelled departments, with one thick padded mailer partway down the stack, and an exception tray. A bank of matching pigeonholes stands within reach, with a parcel shelf below. The robot picks up the first envelope, turns its address toward its head, and places it into the corresponding pigeonhole. It sorts the next envelope the same way, then finds one addressed to another building, pauses to inspect it, and sets it in the exception tray. It reaches the padded mailer and checks it against the intended pigeonhole. Finding it too thick, it withdraws the mailer and places it on the parcel shelf below. It sorts the remaining two envelopes and checks the now-empty input area.

### B07S. A tray transfer parked for an urgent job and resumed

A Unitree G1 humanoid robot transfers parts between two indexed trays at a workshop bench, interrupts its work for an urgent delivery, and then resumes where it stopped. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The trays each have twelve numbered pockets arranged in two rows. At the start, parts one through ten have been transferred to the destination tray, and the robot is lifting part eleven from the source tray. A padded temporary holder sits between the trays, and a separate urgent-job station, with its dispatch cradle and acknowledgement button, stands one step away. An amber lamp illuminates at that station while the robot holds part eleven. It pauses, lowers the part into the temporary holder, and places a small marker beside destination pocket eleven. It turns, steps to the urgent station, picks up a red component, places it in the dispatch cradle, and presses the acknowledgement button. When the lamp goes out, it returns to the bench, checks the marker, retrieves the parked part, and places it into pocket eleven. It removes the marker, transfers part twelve, and checks the empty source tray and completed destination tray.

### B08S. Four drinks carried past a pushed chair

A Unitree G1 humanoid robot carries a tray of drinks across a busy office lounge to a collection table. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. Four partially filled glasses already stand on a rigid serving tray, and the robot has just lifted it with both hands and turned toward the collection table a few steps away. Several people sit at tables along the route, leaving an aisle with one narrow gap between chairs. Take the people from the asset library rather than creating them. The robot walks between the occupied tables until a seated person pushes a chair backward into its path. The robot stops, lets the glasses settle, and waits while the person notices it and pulls the chair inward. It resumes, turns its body sideways to pass through the narrow gap, and brings the tray through at a steady height. At the collection table, it lowers the tray onto the surface and releases its grip.

### B09S. A piano stood upright to clear a newel at the half-landing

Three movers carry an upright piano up a stairwell in an apartment building and negotiate a tight turn at the half-landing. The movers are three men in work clothes. Take them from the asset library rather than creating them. The piano is already wrapped in moving blankets, strapped to a stair-climbing dolly and tipped back, and the movers have just brought it onto the half-landing, with one mover above guiding and two below supporting the load. They begin turning toward the second flight, but the piano's case meets the newel post. They stop, back off slightly, and bring the piano almost upright to make room for the turn. The upper mover changes position while the other two hold the load, then they pivot the piano, reposition the dolly toward the next flight, and tip it back, ready to climb.

### B10S. A coffee finished and a name called twice

A barista finishes a takeaway milk coffee and hands it to a customer in a small café. The barista is a woman in chef's whites and the customer a man in a business suit. Take them from the asset library rather than creating them. The order has already been taken and paid for: the customer's name is written on a paper cup, the portafilter is ground and tamped, and a pitcher of freshly steamed milk stands beside the machine, with lids and sleeves within reach. The customer waits at the collection end of the counter, looking at their phone. The barista locks the portafilter into the machine, sets the cup beneath it and starts the shot, which finishes almost at once. They pour the milk into the cup, press on a lid, fit a sleeve, and call the customer's name. The customer does not respond. The barista glances toward them and calls again, raising the cup slightly. The customer looks up, puts away their phone, and approaches. The barista places the drink within reach, and the customer takes it.

### B11S. A crew's last checks converging on the take

A small film crew makes its final preparations for a dolly shot in a furnished studio interior and shoots the take. Take the crew and the actor from the asset library rather than creating them. A short track is already laid and levelled, the dolly stands at its start with the camera mounted, the lamp and flag are rigged on their stands, and the actor waits beside the set. At the same time, the gaffer takes a light reading beside the actor's position and adjusts the lamp, and the focus puller measures one distance and tapes a mark beside the track. The actor steps onto their floor mark. The director calls for quiet, the camera starts, and an assistant presents and snaps the slate and steps clear. The actor performs as the dolly grip pushes the camera along the track and the focus puller follows the mark, and the dolly comes to rest at the end of its move.

### B12S. Two failed deadlift pulls

A weightlifter attempts a heavy deadlift in a gym and fails to complete it. The lifter is a man in sportswear. Take them from the asset library rather than creating them. The barbell lies on a lifting platform already loaded with two plates on each end and secured with collars; the lifter's belt is tightened and their hands are chalked. They approach the bar, set their feet, bend down, and establish their grip. On the first attempt, they pull the bar to knee height, stall, and lower it back to the platform. They release their grip and stand to take a breath, then change their grip and try again, this time lifting the bar only a short distance before returning it to the floor. They let go and step back.

### B15S. A stay broken at two steps and held on the retry

A trainer teaches a young dog to sit and stay in a quiet fenced garden. The dog is a German shepherd and the trainer a woman in sportswear. Take them from the asset library rather than creating them. The trainer wears a treat pouch, and the dog begins standing in front of them on a clear patch of grass. The trainer holds a treat above the dog’s nose and moves it backward until the dog sits, marks the response with a short word, and gives the treat. They give an open-palm stay signal and take two steps back, but the dog stands and follows. The trainer stops, guides the dog back to its original place, and asks it to sit again. They give the stay signal and take two steps back again, and this time the dog holds its position until the trainer returns. The trainer gives a release cue and rewards the dog as it gets up and approaches.

---

## Medium versions — for a 120-150 s film

Ten versions for a film about twice the short ones' length, run with a
120-150 s film paragraph, alongside B05, B07 and B13-B24 as written. Seven are new cuts that
plan at about 1:40-1:50. B06, B08 and B10 are their full briefs, which already
plan at about that length: B08 verbatim, B06 with "three occupied pigeonholes"
— which contradicts a stack addressed to six departments — made "three of the
pigeonholes", and B10 with the steam wand wiped right after steaming. B05 and
B07 have no medium version, because their full briefs plan at little more than
a minute. B13-B16 cast their animals from the asset library, as in the short
set.

The set delivers within 12-24 hours. Each brief predicts no more than about
150 s of film, and every character is the G1 or comes from the asset library.

| | plan | predicted film | expected runtime |
|---|---|---|---|
| B01M | 1:46 | 2:16 | 12 h (8-16) |
| B02M | 1:34 | 2:00 | 11 h (8-15) |
| B03M | 1:46 | 2:16 | 12 h (8-16) |
| B04M | 1:44 | 2:13 | 12 h (8-16) |
| B06M | 1:33 | 1:59 | 11 h (7-15) |
| B08M | 1:29 | 1:54 | 10 h (7-14) |
| B09M | 1:48 | 2:18 | 12 h (8-17) |
| B10M | 1:51 | 2:22 | 12 h (8-17) |
| B11M | 1:47 | 2:17 | 12 h (8-17) |
| B12M | 1:53 | 2:25 | 12 h (9-17) |
| B05, B07 | 1:00-1:02 | 1:17-1:20 | 8 h (6-11) |
| B13-B16 (library) | 0:44-1:12 | 0:57-1:32 | 7-9 h (5-12) |
| B17-B24 | 0:28-0:52 | 0:36-1:07 | 5-7 h (4-10) |

### B01M. Scrambled eggs from the fridge to the plate

A Unitree G1 humanoid robot cooks scrambled eggs by itself in a compact kitchen. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The preparation counter starts clear. Three eggs in a carton and a portion of butter sit in the refrigerator, a bowl and a whisk stand on a nearby shelf, a frying pan sits on the stovetop, a spatula and a plate are set out beside the stove, and an open waste bin stands beside the counter. The robot opens the refrigerator, takes out the eggs and butter, closes it, and carries them to the counter. It retrieves the bowl and whisk from the shelf, centers the pan on the burner, and turns on the heat. It opens the carton and cracks each egg over the bowl, separating the shells and setting the empty halves aside, then steadies the bowl with one hand and whisks with the other. It drops the butter into the hot pan, where it melts at once, then lifts and tilts the bowl to pour in the eggs and sets the emptied bowl down. The eggs begin to set as soon as they meet the pan, and the robot scrapes and folds them with the spatula into soft curds. It turns off the burner, tilts the pan while guiding the eggs onto the plate, and returns the pan to the stove. Finally, it drops the shells into the bin.

### B02M. An office chair assembled from its opened carton

A Unitree G1 humanoid robot assembles a flat-pack office chair from its opened carton in an open-plan office bay. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The carton stands open on the floor with its flaps folded back. Inside are the five-arm star base, five casters, the gas-lift column with its plastic sleeve, and the seat, which ships with its backrest already attached. A vacant desk stands a few metres away. The robot lifts the star base out of the carton and sets it upside down on the floor, then takes out the casters and the column. It presses each caster into a socket of the upturned base, rights the base, slides the sleeve over the column, and inserts the column into the base hub. It lifts the seat out of the carton, carries it over the column, and lowers it onto the taper. The robot sits briefly to seat the connection, pulls the height lever so the seat sinks under its weight, and leans back to test the backrest tilt. It stands, turns the seat on its swivel by hand, and rolls the finished chair to the desk.

### B03M. The first kiln shelf laid, fixed and loaded

A Unitree G1 humanoid robot lays and loads the first shelf of a front-loading kiln in a pottery studio. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The kiln is cold and its door is open. A nearby trolley holds kiln shelves, support posts of several heights, glazed bowls, small unglazed cups, a larger unglazed bowl, and a clearance gauge. The robot places three posts inside the kiln and lowers the first shelf onto them. It presses lightly at the shelf edges and feels a wobble: one post is chipped at its base and rocks under the shelf. It lifts the shelf, replaces the chipped post with a sound one from the trolley, and checks it again. It loads four glazed bowls with gaps between them, checks the gaps with the gauge, and nests the small unglazed cups inside the larger unglazed bowl on the same shelf.

### B04M. A small break room mopped backward around a table

A Unitree G1 humanoid robot mops a tiled office break room, working backward from the far corner toward the doorway. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. The room is small, about three metres by two and a half, with one table in the middle of the floor and its chairs stacked against a wall, leaving an open floor with several dusty footprints and a small coffee stain. A single doorway leads to a dry hallway, where a folding wet-floor sign already stands beside the entrance. A mop and a wheeled bucket with a wringer stand on the dry floor just doorward of the far corner. The robot dips and wrings the mop and begins in the far corner. It sweeps the mop in overlapping figure-eight strokes, then steps backward onto the remaining dry floor. After several passes, it rinses and wrings the mop and pulls the bucket closer along the dry strip. It works around the table legs, scrubs the coffee stain with shorter strokes, and continues toward the exit as the cleaned wet area expands. Near the doorway, it moves the bucket into the hallway, steps back over the threshold, and mops the final strip from the hallway. It lifts the mop clear, secures it in the bucket, and leaves the warning sign beside the freshly cleaned room.

### B06M. Mail sorted with exceptions and bundled (full brief, one premise corrected)

A Unitree G1 humanoid robot sorts a mixed bundle of mail at a mailroom sorting station. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. A counter holds a stack of twelve envelopes addressed to six clearly labelled departments, an exception tray, rubber bands, and one small parcel. A bank of matching pigeonholes stands within reach, with a parcel shelf below. The robot squares the bundle, picks up the first envelope, turns its address toward its head, and places it into the corresponding pigeonhole. It continues through several envelopes at a steady pace until it finds one addressed to another building, pauses to inspect it, and sets it in the exception tray. It resumes sorting, then reaches the parcel and checks it against the intended pigeonhole. Finding it too large, it withdraws the parcel and places it on the lower shelf. It works through the remaining envelopes, sets aside a second misdirected item, and checks the now-empty input area. Finally, it gathers the mail from three of the pigeonholes into separate bundles, secures each with a rubber band, and places them in a collection tray, leaving the exceptions separate.

### B08M. Carrying a loaded tray through a crowded room (full brief, people from the asset library)

A Unitree G1 humanoid robot carries a tray of drinks from a service counter to a collection table across a busy office lounge. The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the asset library rather than creating a new robot. Six partially filled glasses stand beside a rigid serving tray on the counter. Several people sit at tables along the route, leaving an aisle with one narrow gap between chairs. Take the people from the asset library rather than creating them. The robot places the glasses onto the tray, lifts it with both hands, and turns toward the collection table. It walks between the occupied tables until a seated person pushes a chair backward into its path. The robot stops, lets the glasses settle, and waits while the person notices it and pulls the chair inward. It resumes, turns its body sideways to pass through the narrow gap, and brings the tray through at a steady height. Beyond the gap, a passing person lightly bumps its upper arm. The robot pauses, bends its knees slightly, adjusts its stance, and steadies the tray before continuing. At the collection table, it lowers the tray onto the surface, releases its grip, and arranges the six glasses one by one on coasters. It then picks up the empty tray and returns toward the service counter.

### B09M. A piano walked up a flight and round a newel

Three movers carry an upright piano up a stairwell in an apartment building, negotiating a tight turn at the half-landing. The movers are three men in work clothes. Take them from the asset library rather than creating them. The piano stands at the foot of the stairs, already wrapped in moving blankets and strapped to a stair-climbing dolly. The movers tip it backward together, with one mover above guiding and two below supporting the load. They find their balance, approach the first step, and work up the first flight one step at a time, pausing midway to adjust their grips. At the half-landing, they begin turning toward the second flight, but the piano's case meets the newel post. They stop, back off slightly, and bring the piano almost upright to make room for the turn. The upper mover changes position while the other two hold the load, then they pivot the piano and reposition the dolly toward the next flight. They tip it back again and climb the first four steps of the second flight.

### B10M. A coffee order from arrival to handoff (full brief, one premise corrected)

A barista serves a customer from ordering to collecting a takeaway coffee in a small café. The barista is a woman in chef's whites and the customer a man in a business suit. Take them from the asset library rather than creating them. The counter contains a till, paper cups, a marker, an espresso machine, a grinder, a milk pitcher, lids, sleeves, and a cleaning cloth. The customer approaches the till, orders a milk coffee, and gives their name. The barista marks a cup, repeats the order, and takes payment while the customer moves to the collection end of the counter and looks at their phone. The barista grinds and tamps the coffee, locks the portafilter into the machine, and starts the shot. While it extracts, they pour milk into the pitcher, steam it, and wipe and purge the steam wand, then remove the cup and pour in the milk. They press on a lid, fit a sleeve, and call the customer’s name. The customer does not respond. The barista glances toward them and calls again, raising the cup slightly. The customer looks up, puts away their phone, and approaches. The barista places the drink within reach, the customer takes it and leaves, and the barista wipes the counter before returning the pitcher to its place.

### B11M. A dolly shot rigged, rehearsed, fixed and taken

A small film crew finishes setting up a dolly shot in a furnished studio interior, rehearses it, and shoots the take. Take the crew and the actor from the asset library rather than creating them. The track sections are already laid end to end, joined and levelled with wedges, the lamp and flag are rigged on their stands, and the camera dolly, camera, tape and a light meter wait beside the track. Two crew members lift the dolly onto the rails and mount the camera. The gaffer takes a light reading beside the actor's position and adjusts the lamp while the focus puller measures two distances and tapes marks beside the track. An actor steps onto their floor mark for a rehearsal. The dolly grip pushes the camera through the planned move, the focus puller follows the distance marks, and the crew stops to correct a bump at one track joint. They return the dolly to its starting position and make a final lighting check. The director calls for quiet, the camera starts, an assistant presents and snaps the slate, and the actor performs as the dolly travels along the track.

### B12M. A failed deadlift and the bar stripped

A weightlifter attempts a heavy deadlift in a gym, fails to complete it, and puts the equipment away. The lifter is a man in sportswear. Take them from the asset library rather than creating them. The barbell lies on a lifting platform beside a plate rack, already loaded with two plates on each end and secured with collars; the lifter's belt is tightened and their hands are chalked. They approach the bar, set their feet, bend down, and establish their grip. On the first attempt, they pull the bar to knee height, stall, and lower it back to the platform. They release their grip, stand to take a breath, then reset their feet and try again. The second pull rises slightly higher but still stops before lockout, and they lower it once more. After a longer pause, they change their grip and make a third attempt, this time lifting the bar only a short distance before returning it to the floor. They let go, step back, and loosen their belt. Finally, they remove the collars and unload the plates one at a time onto the rack, leaving the empty bar on the platform.

---

## Appendix — staged ids

The id each brief is staged under, which names its run and its film.

| | staged id |
|---|---|
| B01 | `g1_scrambles_eggs` |
| B02 | `g1_assembles_office_chair` |
| B03 | `g1_loads_pottery_kiln` |
| B04 | `g1_mops_break_room` |
| B05 | `g1_carries_panel_through_doorway` |
| B06 | `g1_sorts_mailroom_post` |
| B07 | `g1_transfers_parts_after_interrupt` |
| B08 | `g1_carries_drinks_tray` |
| B09 | `movers_carry_piano_upstairs` |
| B10 | `barista_serves_takeaway_coffee` |
| B11 | `film_crew_sets_dolly_shot` |
| B12 | `lifter_fails_heavy_deadlift` |
| B13 | `dog_fetches_newspaper` |
| B14 | `crow_opens_puzzle_feeder` |
| B15 | `trainer_teaches_dog_to_stay` |
| B16 | `macaque_cracks_nut` |
| B17 | `marble_course_runs` |
| B18 | `domino_circuit_falls` |
| B19 | `parcel_sorter_sorts_six` |
| B20 | `tipping_bucket_fills_tank` |
| B21 | `bottling_line_fills_six` |
| B22 | `drawbridge_opens_for_boat` |
| B23 | `spring_cart_runs_course` |
| B24 | `clock_strikes_four` |
