#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Set

# Synonym table adapted from the local CHAIR implementation.
SYNONYMS_TXT = """
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, foreigner, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, goat, ewe
cow, cattle, oxen, ox, calf, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iphone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
"""


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def simple_singular(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ses") and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize_term(term: str) -> str:
    term = normalize_space(term)
    parts = [simple_singular(p) for p in term.split(" ")]
    return " ".join(parts)


def build_synonym_map(coco_categories):
    canonical = {normalize_term(c["name"]) for c in coco_categories}
    alias_to_canonical = {}

    for raw_line in SYNONYMS_TXT.strip().splitlines():
        words = [normalize_term(x) for x in raw_line.split(",") if x.strip()]
        if not words:
            continue

        head = words[0]
        if head in canonical:
            target = head
        else:
            target = None
            for w in words:
                if w in canonical:
                    target = w
                    break
            if target is None:
                continue

        alias_to_canonical[target] = target
        for w in words:
            alias_to_canonical[w] = target

    for c in canonical:
        alias_to_canonical[c] = c

    # Disambiguation for common phrases.
    alias_to_canonical["bow tie"] = "tie"
    alias_to_canonical["toilet seat"] = "toilet"

    return alias_to_canonical, canonical


def extract_objects(caption: str, alias_to_canonical):
    text = normalize_space(caption)
    if not text:
        return []

    normalized_text = normalize_term(text)
    found = []
    # Match longer aliases first to avoid splitting multi-word objects.
    aliases = sorted(alias_to_canonical.keys(), key=lambda x: len(x), reverse=True)
    for alias in aliases:
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, normalized_text):
            found.append(alias_to_canonical[alias])

    # Deduplicate while preserving order.
    dedup = []
    seen = set()
    for obj in found:
        if obj not in seen:
            dedup.append(obj)
            seen.add(obj)
    return dedup


def load_gt_objects(instances_path: Path, valid_image_ids: Set[int]):
    data = json.loads(instances_path.read_text())
    categories = data["categories"]
    cat_id_to_name = {int(c["id"]): normalize_term(c["name"]) for c in categories}

    gt_objects = defaultdict(set)
    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in valid_image_ids:
            continue
        cat_name = cat_id_to_name[int(ann["category_id"])]
        gt_objects[image_id].add(cat_name)

    return categories, gt_objects


def main():
    parser = argparse.ArgumentParser(description="Compute CHAIR metrics using COCO instances_val2014 annotations.")
    parser.add_argument("--captions-json", required=True, help="Path to generated captions JSON list.")
    parser.add_argument("--instances-json", required=True, help="Path to COCO instances_val2014.json.")
    parser.add_argument("--analysis-json", required=True, help="Path to save per-image hallucination analysis.")
    args = parser.parse_args()

    captions = json.loads(Path(args.captions_json).read_text())
    image_ids = {int(x["image_id"]) for x in captions}

    categories, gt_objects = load_gt_objects(Path(args.instances_json), image_ids)
    alias_to_canonical, canonical_vocab = build_synonym_map(categories)

    total_mentioned = 0
    total_hallucinated = 0
    total_sentences = 0
    hallucinated_sentences = 0
    total_gt_objects = 0
    total_recalled_objects = 0

    details = []
    for row in captions:
        image_id = int(row["image_id"])
        caption = row["caption"]

        pred_objects = extract_objects(caption, alias_to_canonical)
        pred_objects = [x for x in pred_objects if x in canonical_vocab]
        gt = sorted(gt_objects.get(image_id, set()))
        gt_set = set(gt)

        hallucinated = sorted([x for x in pred_objects if x not in gt_set])
        recalled_objects = sorted([x for x in pred_objects if x in gt_set])
        total_gt_objects += len(gt_set)
        total_recalled_objects += len(recalled_objects)

        if pred_objects:
            total_mentioned += len(pred_objects)
            total_hallucinated += len(hallucinated)

        has_hall = len(hallucinated) > 0
        total_sentences += 1
        if has_hall:
            hallucinated_sentences += 1

        details.append(
            {
                "image_id": image_id,
                "caption": caption,
                "predicted_objects": pred_objects,
                "gt_objects": gt,
                "hallucinated_objects": hallucinated,
                "recalled_objects": recalled_objects,
                "has_hallucination": has_hall,
            }
        )

    chair_i = (total_hallucinated / total_mentioned) if total_mentioned > 0 else 0.0
    chair_s = (hallucinated_sentences / total_sentences) if total_sentences > 0 else 0.0
    recall = (total_recalled_objects / total_gt_objects) if total_gt_objects > 0 else 0.0

    report = {
        "overall_metrics": {
            "CHAIRi": chair_i,
            "CHAIRs": chair_s,
            "Recall": recall,
            "num_total_sentences": total_sentences,
            "num_sentences_with_hallucination": hallucinated_sentences,
            "num_total_mentioned_objects": total_mentioned,
            "num_hallucinated_objects": total_hallucinated,
            "num_total_gt_objects": total_gt_objects,
            "num_recalled_objects": total_recalled_objects,
        },
        "sentences": details,
    }

    Path(args.analysis_json).write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"CHAIR-i: {chair_i:.6f}")
    print(f"CHAIR-s: {chair_s:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"Details saved to: {args.analysis_json}")


if __name__ == "__main__":
    main()
