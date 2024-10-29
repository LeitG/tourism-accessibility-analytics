import json
from collections import defaultdict

with open('disability_reviews_all.json', 'r') as file:
    reviews = json.load(file)

reviews_by_author = defaultdict(list)
for review in reviews:
    author_id = review['author_id']
    reviews_by_author[author_id].append(review)

train_reviews = []
test_reviews = []
for author_id, author_reviews in reviews_by_author.items():
    if len(author_reviews) >= 5:
        author_reviews.sort(key=lambda x: x['created'])
        test_reviews.append(author_reviews[-1])
        train_reviews.extend(author_reviews[:-1])

with open('filtered_reviews_train.json', 'w') as file:
    json.dump(train_reviews, file, indent=4)

with open('filtered_reviews_test.json', 'w') as file:
    json.dump(test_reviews, file, indent=4)

print(f'Train set contains {len(train_reviews)} reviews')
print(f'Test set contains {len(test_reviews)} reviews')
