def detect_growth_events(df, threshold=-350, area_label='area (sq µm)'):
    """Given a dataframe containing bacterial growth areas, detect growth events.

    A growth event is defined as when the bacterial area changed by a certain
    threshold value.

    :param df: dataframe of bacterial areas
    :type df: pandas.DataFrame
    :param threshold: the bacterial area change threshold used to detect
                      growth events, defaults to -350
    :type threshold: int
    :param area_label: dataframe column label for the bacterial area,
                       defaults to 'area (sq µm)'
    :type area_label: str, optional

    :return: list containing growth events
    :rtype: list
    """
    event_id = 0
    events = []
    for ele in df[area_label].diff() < threshold:
        if ele:
            event_id += 1
        events.append(event_id)
    return events
