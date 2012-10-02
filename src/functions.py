def get_expected_class(subject, trial_index, scan_index):
    '''
    Checks the subject's info to see what class is expected when
    classifying a scan from a certain trial. Trial index runs from 1 to 54.
    Since we only know 'firstStimulus' for a trial, we'll map all scans before
    [...] to firstStimulus and the rest to the secondStimulus.

    According to data, trial_index 1..27 all have firstStimulus = 'P'.
    Each trial has ~54 scans, so we'll asume that 27 is the halfwayPoint/

    Returns either 'Picture' or 'Sentence'
    '''
    halfwayPoint = 27
    first_stimulus = subject['info'][0][trial_index]['firstStimulus'][0]
    if first_stimulus == 'P':
        if scan_index < halfwayPoint:
            return 'Picture'
        else:
            return 'Sentence'
    else:
        if scan_index < halfwayPoint:
            return 'Sentence'
        else:
            return 'Picture'
