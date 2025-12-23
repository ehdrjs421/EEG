def extract_seizure_events(y_true):
    events = []
    in_event = False

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_event:
            start = i
            in_event = True
        elif y_true[i] == 0 and in_event:
            events.append((start, i - 1))
            in_event = False

    if in_event:
        events.append((start, len(y_true) - 1))

    return events
