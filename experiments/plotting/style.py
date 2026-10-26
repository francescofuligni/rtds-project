"""Poster plot styles shared by feature and selection figures."""

STYLES = {
    "none": ("#555555", "o", "No background"),
    "A": ("#009E73", "s", "A: marital status"),
    "B": ("#D55E00", "^", "B"),
    "A+B": ("#0072B2", "D", "A+B"),
}


SELECTION_STYLES = {
    "random": ("#0072B2", "o", "Random"),
    "sequential": ("#D55E00", "s", "Sequential"),
}

BACKGROUND_STYLES = {
    "none": ("#555555", "o", "None (education run)"),
    "education": ("#0072B2", "D", "Marital + education"),
    "age": ("#009E73", "s", "Marital + age"),
    "uniform": ("#888888", "o", "Uniform baseline"),
    "marginal": ("#7B5AA6", "o", "Marginal baseline"),
}
