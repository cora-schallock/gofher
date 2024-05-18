from classify import get_opposite_label

def score_label(label,paper_label):
    opposite = get_opposite_label(label)

    correct_label_letter_count = len(set([*label.lower()]).union([*paper_label.lower()]))
    incorrect_label_letter_count = len(set([*opposite.lower()]).union([*paper_label.lower()]))
        
    if correct_label_letter_count > incorrect_label_letter_count and correct_label_letter_count > 1:
        return 1
    elif incorrect_label_letter_count > correct_label_letter_count and incorrect_label_letter_count > 1:
        return -1
    return 0