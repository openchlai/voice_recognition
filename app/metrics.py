from jiwer.measures import cer, mer, wer, wil, wip

def metrics(text, hypo):
    pass
    """
    Calculate Jiwer Metric between one or more `reference` 
    and `hypothesis` sentences.
    """

    data = {}

    try:
        
        """Word Error Rate (WER)"""
        data['wer'] = wer(text, hypo)

        """Match Error Rate (MER)"""
        data['mer'] = mer(text, hypo)

        """Word Information Lost (WIL)"""
        data['wil'] = wil(text, hypo)

        """Word Information Preserved (WIP)"""
        data['wip'] = wip(text, hypo)

        """Character Error Rate (CER)"""
        data['cer'] = cer(text, hypo)

    except Exception as e:
        data['error'] = " {}".format(e)

    return data


if __name__ == '__main__':
    text = "it's electronic sabotage when nations track payment"  # Reference Txt
    hypo = "the electronic subterfuge is when nations track payment"  # Hypothesis

    print(metrics(text, hypo))
