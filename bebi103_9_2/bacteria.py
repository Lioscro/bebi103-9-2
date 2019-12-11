import numpy as np

def detect_growth_events(areas, threshold=-350):
    """Given a list of bacterial growth areas, detect growth events.

    A growth event is defined as when the bacterial area changed by a certain
    threshold value.

    :param areas: list of areas
    :type areas: list
    :param threshold: the bacterial area change threshold used to detect
                      growth events, defaults to -350
    :type threshold: int

    :return: list containing growth events
    :rtype: list
    """
    event_id = 0
    events = [event_id]
    for ele in np.diff(areas) < threshold:
        if ele:
            event_id += 1
        events.append(event_id)
    return events

def normalize_times(times, events):
    """Given a list of times and event ids, normalize the times (i.e. calculate
    the time since the start of each event).

    :param times: list of times
    :type times: list
    :param events: list of event ids
    :type events: list

    :return: list of normalized times
    :rtype: list
    """
    normalized_times = []
    for event in sorted(np.unique(events)):
        event_times = times[events == event]
        normalized_times.extend(list(event_times - min(event_times)))
    return normalized_times
