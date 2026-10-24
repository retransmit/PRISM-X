"""The live loop (design section 9): Prism learns from its own traffic.

The audit log keeps what was scored and the labels the bank learned; a challenger is
trained on them, backtested against the serving champion and promoted only on
evidence; drift of the inputs and of the score is measured against each bundle's own
training reference.
"""
