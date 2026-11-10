# Animation briefs with imported characters

These are the character briefs of the final benchmark set (B10, B13, B14, B15, B16) that use imported
character rigs instead of characters the agent authors from scratch. Each one is the brief from
`animation_24_briefs.md`, unchanged except for one added sentence, placed after the first.

**That sentence describes the characters and the agent retrieves them.** It names who the
characters are ("the dog is a German shepherd") and tells the agent to take them from the asset library,
with no path. A DyCraft arm finds them in `references/assets/INDEX.md` (`assets/imported/`, cataloged in
`catalogs/asset_library.json`) and places them with `load_asset("<bundle>")`. The bare arm is never given
these variants or the library (`benchmark/bare.py`): the paper's bare runs received the briefs as written in
`animation_24_briefs.md`. The bundles the descriptions match:

| brief | characters | bundles in the library |
|---|---|---|
| B10 | barista, customer | `rb_chef_female_01_rig`, `rb_business_male_01_rig` |
| B13 | dog, owner | `sf_dog_gsd_puppyclub_rig`, `rb_male_adult_03_rig` |
| B14 | crow | `crow_rig` |
| B15 | dog, trainer | `sf_dog_gsd_puppyclub_rig`, `rb_sports_female_02_rig` |
| B16 | macaque | none; `sf_chimp_sfm_rig` is the closest primate |

The human rigs carry nine anatomical joint limits each. The dog and the crow carry none, because their
skeletons are not a recognized human family. The chimpanzee standing in for the macaque gets the human limits.

### B10. A coffee order from arrival to handoff

A barista serves a customer from ordering to collecting a takeaway coffee in a small café. Take the characters from the asset library rather than creating them: the barista is a woman in chef's whites and the customer a man in a business suit. The counter contains a till, paper cups, a marker, an espresso machine, a grinder, a milk pitcher, lids, sleeves, and a cleaning cloth. The customer approaches the till, orders a milk coffee, and gives their name. The barista marks a cup, repeats the order, and takes payment while the customer moves to the collection end of the counter and looks at their phone. The barista grinds and tamps the coffee, locks the portafilter into the machine, and starts the shot. While it extracts, they pour milk into the pitcher and steam it, then remove the cup and pour in the milk. They press on a lid, fit a sleeve, and call the customer’s name. The customer does not respond. The barista glances toward them and calls again, raising the cup slightly. The customer looks up, puts away their phone, and approaches. The barista places the drink within reach, the customer takes it and leaves, and the barista wipes the steam wand and counter before returning the pitcher to its place.

### B13. A dog fetches the newspaper

A dog retrieves a rolled newspaper from the front garden and delivers it to its owner on the porch. Take the characters from the asset library rather than creating them: the dog is a German shepherd and its owner an older man. The owner sits beside the front door, and a newspaper secured with a paper band lies near the garden gate at the end of a short path. The dog rests beside the chair until the owner gives a verbal cue and points toward the gate. It raises its head, stands, walks to the porch steps, and trots down onto the path. Near the gate, it slows, lowers its nose, and locates the newspaper beside a plant pot. It grips the middle of the roll in its mouth, lifts it, turns, and trots back toward the house. At the steps, it slows and climbs while carrying the newspaper. On reaching the porch, it pauses to adjust its grip, then approaches the owner and raises its muzzle toward the waiting hand. The owner takes hold of the newspaper before the dog releases it. The dog sits beside the chair, the owner places the newspaper on their lap and strokes its head, and the dog settles at their feet.

### B14. A crow opens a puzzle feeder

A crow works out how to open a puzzle feeder on a low outdoor table. Take the crow from the asset library rather than creating a new bird. Food is visible behind a transparent sliding hatch, but a removable peg blocks the hatch from opening. A perch stands beside the table. The crow lands near the feeder, walks around it, and tilts its head to inspect the food through the clear panel. It pecks the panel several times, then catches the edge of the hatch with its bill and pushes it sideways. The hatch moves a little and stops against the peg. The crow steps back, looks along the hatch edge, and pecks at the protruding peg. It grips the peg with its bill and pulls, but loses its grip. It changes position, braces beside the feeder, hooks one foot around the peg, and draws it out of the retaining hole. The peg drops onto the table. The crow returns its bill to the hatch edge and slides the hatch far enough to expose the food. It reaches inside, takes a piece, withdraws its head, and hops onto the nearby perch to eat, leaving the hatch open and the peg on the table.

### B15. Teaching a dog to sit and stay

A trainer teaches a young dog to sit and stay in a quiet fenced garden. Take the characters from the asset library rather than creating them: the dog is a German shepherd and the trainer a woman in sportswear. The trainer wears a treat pouch, and the dog begins standing in front of them on a clear patch of grass. The trainer holds a treat above the dog’s nose and moves it backward until the dog sits, marks the response with a short word, and gives the treat. They repeat the sit with a raised-hand signal, reward it, and then introduce an open-palm stay signal. The trainer steps backward one pace, pauses, returns to the dog, and rewards it while it remains seated. On the next attempt, the trainer takes two steps back, but the dog stands and follows. The trainer stops, guides the dog back to its original place, and asks it to sit again. They repeat the easier one-step stay and reward it, then try two steps again. This time the dog holds its position until the trainer returns. Finally, the trainer backs away three paces and briefly turns sideways before facing the dog again. The dog remains seated. The trainer returns, gives a release cue, and rewards the dog as it gets up and approaches.

### B16. A young macaque learns to crack a nut

A young macaque attempts to crack a hard-shelled nut on a flat stone in a forest clearing. Take the macaque from the asset library, or the closest primate it offers, rather than creating a new animal. Several nuts lie beside the stone, with a small cobble nearby and a larger, heavier stone a short distance away. The macaque picks up a nut, places it on the flat surface, and lifts the small cobble in one hand. It strikes the nut, but the nut rolls off the stone without opening. The macaque follows it, retrieves it, and places it back near the center. A second strike sends the nut skidding to the edge, still intact. It retrieves the nut again, sets down the small cobble, and moves to the heavier stone. It lifts that stone with both hands, brings it over the nut, and strikes, but misses to one side. The macaque lowers the stone, repositions the nut, widens its stance, and adjusts its grip. It raises the stone again and strikes squarely, cracking the shell. It sets the stone down beside the anvil, separates the shell pieces with its fingers, picks out the kernel, and sits beside the scattered fragments to eat.
