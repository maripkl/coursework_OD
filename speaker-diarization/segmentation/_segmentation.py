from pyannote.core import Annotation, Timeline, Segment, SlidingWindow


def split_segments(vad, win_size, step_size):
    segments_speech = []
    for segment in vad:
        start = segment.start
        end = segment.end
        duration = end - start
        if duration < 0. * win_size:
            pass
            # ignore very short segments
        elif duration < win_size + step_size:
            segment_short = segment
            segments_speech += [segment_short]
        else:
            sliding_window = SlidingWindow(win_size, step_size)
            for segment_short in sliding_window(segment):
                segments_speech += [segment_short]      
            segments_speech[-1] = Segment(segment_short.start, segment.end)
    return segments_speech


def get_overlap_timeline(reference):
    overlap = Timeline(uri=reference.uri)
    for (s1, t1), (s2, t2) in reference.co_iter(reference):
        l1 = reference[s1, t1]
        l2 = reference[s2, t2]
        if l1 == l2:
            continue
        overlap.add(s1 & s2)
    return overlap.support()


def split_segment_in2(segment):
    return Segment(segment.start, segment.middle), Segment(segment.middle, segment.end)


def split_overlap_part(reference):
    overlap = get_overlap_timeline(reference) # overlap timeline
    to_crop = overlap.gaps(reference.get_timeline().extent()) # inverse of the overlap timeline
    res = reference.crop(to_crop)
    for seg in overlap:
        first, second = split_segment_in2(seg)
        res[first] = reference.crop(Segment(first.start-0.01, first.start)).labels()[0]
        res[second] = reference.crop(Segment(second.end, second.end+0.01)).labels()[0]
    return res