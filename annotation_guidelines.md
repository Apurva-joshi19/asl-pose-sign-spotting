# Annotation guidelines (version g1)

Annotator: single hearing non-signer. Reference: ASLLRP Sign Bank videos (https://dai.cs.rutgers.edu/dai/s/signbank).
Lexicon: `lexicon.csv` (closed list; version recorded per clip as `lexicon_version`).

## 1. What to label
- Label ONLY signs from the lexicon. Everything else (other signs, fingerspelling, classifiers, gestures,
  mouthing, pauses) is NOT labelled; CTC's blank absorbs it. There is no <OTHER> token.
- A sign may be marked only if it is visually matched to its Sign Bank reference video.
- English is used ONLY to choose clips and to pre-list candidate signs in the tool.
  Never add a label because the English sentence contains the word; never skip a check because it does not.

## 2. When a sign counts as a match
- Mark the sign as **sure (S)** only if handshape, location AND movement all match the reference.
  Normal variation in speed and size is allowed.
- If one of the three is doubtful, or the sign is partly hidden or blurred, mark it **unsure (U)**.
- If you are not even unsure (it is only a vague resemblance), do not mark it.
- 1-handed and 2-handed forms of the same sign are the same gloss_id.
- Directional forms listed in `accepted_variants` (e.g. HELP toward self) are the same gloss_id.
- Compounds (e.g. KNOW+NEG, NOW+DAY, LEARN+AGENT) are NOT the lexicon sign; do not mark them.

## 3. Peak frame
- Peak frame = the frame where the handshape is fully formed at the sign's main location.
- Frame numbers are the tool's frame indices: 0-based, in the clip's own video frames
  (clips are 24 or 60 fps; the fps is stored per clip, never assume 30).
- Repeats: give separate marks only if the movements are clearly separate signs (hands return/pause in between).
  A sign with built-in repetition (e.g. a double tap) is ONE mark.

## 4. Pointing
- Only IX-1p is labelled: index finger pointing to/touching the signer's own chest.
- Flat hand on chest (POSS-1p, "my") is NOT IX-1p. Pointing to other places/people is not labelled.

## 5. Gloss order
- The sequence is the order of peak frames in the video (the tool sorts automatically), never English word order.

## 6. Procedure per clip
1. Play the clip once at 1x.
2. Scrub frame by frame through the whole clip.
3. Check every pre-listed candidate sign (from the English sentence) against its reference video.
4. Scan for the 15 most frequent lexicon signs, even if not pre-listed:
   GO, WANT, LITTLE-BIT, THING, USE, GOOD, COME, LOOK, TIME, MUST, KNOW, START, MANY, TALK, IX-1p.
5. Add marks (gloss_id, peak frame, S/U); save. The tool logs minutes per clip.
- If no lexicon sign is found, save the clip with an empty sequence (this is a valid result).

## 7. Blind mode
- Used for calibration (ASLLRP utterances) and for the second-pass re-annotation (self-consistency).
- Existing labels and the English sentence are hidden; use full-lexicon search instead of pre-listed candidates.
- Second pass happens at least 2 days after the first pass of that clip.

## 8. Honesty rules
- Never edit a first-pass label after seeing the second pass, except test-set disagreements resolved
  against the reference video and logged in the resolution log.
- Never describe these labels as expert-verified or Deaf-verified.
