import copy

from adapter import canonical_ssf
from exp import Exp_Online


class Exp_CanonicalSSF(Exp_Online):
    """
    Online phase for CanonicalSSF (raw learnable out-scale + shift, no drift,
    no hypernetwork -- see adapter/canonical_ssf.py).

    Unlike Exp_Proceed, no freeze_adapter/freeze_bias alternation is needed:
    there is no weight-vs-bias split to guard against -- scale/shift are always
    trainable and backbone is always frozen, so the generic recent-batch-update
    / current-batch-forward loop already implemented in Exp_Online (built to
    tolerate steps with nothing trainable, via exp_basic.py's backward guard)
    is exactly the training procedure this module needs. That is why this class
    only overrides _build_model.

    Runs val -> test -> online (same three online_phases as Exp_Proceed) so the
    comparison to a frozen-PROCEED run isolates hypernetwork+drift as the only
    difference: both start the online/test phase from an adapter primed on the
    same validation-set update.
    """

    def __init__(self, args):
        args = copy.deepcopy(args)
        # Required by add_ssf_adapters_ (mirrors Exp_Proceed forcing this for
        # add_down_up_): the learnable-SSF forward path always takes the
        # scale.shape[0] == 1 merge fast path, so weights must actually be merged.
        args.merge_weights = 1
        super(Exp_CanonicalSSF, self).__init__(args)
        self.online_phases = ['val', 'test', 'online']

    def _build_model(self, model=None, framework_class=None):
        model = super()._build_model(model, framework_class=canonical_ssf.CanonicalSSF)
        print(model)
        return model
