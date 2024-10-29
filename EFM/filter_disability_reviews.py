import json

disability_keywords = [
    'wheelchair',
    'accessible',
    'disability',
    'disabled',
    'accessibility',
    'accessible rooms',
    'accessible bathroom',
    'roll-in shower',
    'grab bars',
    'handicap accessible',
    'braille',
    'hearing accessible',
    'visual alarms',
    'wheelchair ramp',
    'elevator',
    'mobility',
    'service animal',
    'guide dog',
    'assistance dog',
    'lowered counter',
    'TTY',
    'TDD',
    'ADA compliance',
    'inclusive',
    'special needs',
    'adapted room',
    'hearing impaired',
    'visual impairment',
    'physical disability',
    'mobility impaired',
    'wheelchair lift',
    'accessible parking',
    'accessible transit',
    'accessible path of travel',
    'step-free access',
    'automatic door',
    'handicap room',
    'mobility scooter',
    'mobility aid',
    'prosthesis',
    'orthotic',
    'crutches',
    'walker',
    'sign language',
    'communication aid',
    'accessible seating',
    'priority seating',
    'ramp access',
    'handicap facilities',
    'special accommodation',
    'support animal',
    'accessible shower',
    'bathtub bench',
    'bathtub seat',
    'handheld shower',
    'shower chair',
    'transfer bench',
    'accessible sink',
    'accessible toilet',
    'toilet rails',
    'doorway width',
    'clearance space',
    'raised toilet',
    'lowered peephole',
    'lowered bed',
    'vibrating alarm',
    'bed shaker alarm',
    'closed captioning',
    'TTY/TDD kit',
    'volume control telephones',
    'visual fire alarm',
    'visual door knock',
    'visual phone call',
    'auditory guidance',
    'sensory friendly',
    'disability friendly',
    'accommodating disability',
    'special equipment',
    'customized facilities',
    'adapted facilities'
]


def load_reviews(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]


def contains_disability_keywords(text, keywords):
    return any(keyword in text.lower() for keyword in keywords)


def save_reviews(reviews, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(reviews, file, ensure_ascii=False, indent=2)


def main():
    reviews_data = load_reviews('reviews_test.json')

    disability_user_ids = set()

    disability_reviews = []

    num = 0
    for review in reviews_data:
        num += 1
        if contains_disability_keywords(review['text'], disability_keywords):
            disability_user_ids.add(review['author_id'])
            disability_reviews.append(review)
        if num % 1000 == 0:
            print(num, len(disability_user_ids), len(disability_reviews))

    save_reviews(disability_reviews, 'disability_reviews.json')

    if disability_user_ids:
        disability_reviews = [review for review in reviews_data if review['author_id'] in disability_user_ids]
        save_reviews(disability_reviews, 'disability_reviews_all.json')


if __name__ == '__main__':
    main()
