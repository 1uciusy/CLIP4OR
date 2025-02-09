from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./testresults/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--prob_map', action='store_true', help='chooses outputting prob maps or binary predictions')
        parser.add_argument('--no_label', action='store_true', help='chooses if we have gt labels in testing phase')
        parser.add_argument("--extra_v_encoder", action='store_true', help="whether use extra vision encoder")
        parser.add_argument("--fix_v_encoder", action='store_true', help="whether stop gradient on vision encoder")
        parser.add_argument("--pretrain_model_path", type=str, default="",
                            help="path where the pretrained model is saved, if not assigned, use ImageNet pretrained model")
        parser.add_argument("--only_extra_v_encoder", action='store_true',
                            help="whether only use external vision encoder")
        self.isTrain = False
        return parser
