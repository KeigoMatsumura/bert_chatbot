# 設定をconfig.json から読み込み，JSONの辞書変数をオブジェクト変数に変換
import json
from turtle import forward, pos

config_file = "./weights/bert_config.json"
 
# ファイルを開き，JSONとして読み込む
json_file = open(config_file, 'r')
config = json.load(json_file)

# 出力確認
print(config)


# 辞書変数をオブジェクト変数に
from attrdict import AttrDict

config = AttrDict(config)
print(config.hidden_size)


# BERT用にLayerNormalization 層を定義する
# 実装の細かな点をTensorFlowに合わせる

class BertLayerNorm(nn.Modlue):
    """LayerNormalization層"""

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size)) # wightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size)) # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BertEmbeddings(nn.Module):
    """文章の単語ID列と，１文目か２文目かの情報を，埋め込みベクトルに変換する"""

    def __init__(self, config):
        super(BertEmbeddings,self).__init__()

        # ３つのベクトル表現の埋め込み

        # Token Embedding: 単語IDを単語ベクトルに変換
        # vocab_size = 30522でBERTの学習済みモデルで使用したボキャブラリの量
        # hidden_size = 768 で特徴量ベクトルの長さは768

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )

        # padding_idx=0はidx=0 の単語ベクトルは０にする
        # BERTのボキャブラリのidx=0が[PAD]である

        # Transformer Positional Embedding: 位置情報テンソルをベクトルに変換
        # Transfomreの場合はsin, cos からなる固定値だったが，BERTはがくしゅうさせる
        # max_position_embeddings = 512 で文の長さは512単語

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Sentence Embedding: 文章の１文目，２文目の情報をベクトルに変換
        # type_vocab_size = 2
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # 作成したLayerNormalization 層
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Dropout 'hidden_dropoput_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, token_type_ids=None):
        """input_ids:[batch_size, seq_len]の文章の単語IDの羅列
        token_type_ids:[batch_size, seq_len]の各単語が１文目なのか，２文目なのかを示すid
        """

        # 1.Token Embeddings
        # 単語IDを単語ベクトルに変換
        words_embeddings = self.word_embeddings(input_ids)


        # 2.Sentence Embedding
        # token_type_ids がない場合は文章の全単語を１文目として，０にする
        # そこで，input_idsと同じサイズのゼロテンソルを作成
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 3.Transformer Positional Embedding:
        # [0,1,2,....]と文章のながさだけ，数字が１ずつ昇順に入った
        # [batch_size, seq_len]のテンソルposition_idsを作成
        # position_idsをにゅうりょくして，position_embeddings 層から768次元のテンソルを取り出す
        seq_length = input_ids.size(1) # 文章の長さ
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # 3つの埋め込みテンソルを足し合わせる[batch_size, seq_len, hidden_size]
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # LayerNormalization とDropoutを実行
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Bertlayer(nn.Module):
    '''BERTのBertLayerモジュール．Transformer'''
    def __init__(self, config):
        super(Bertlayer, self).__init__()

        # Self-Attentionの部分
        self.attention = BertAttention(config)

        # Self-Attentionの出力を処理する全結合層
        self.intermediate = BertIntermediate(config)

        # self-Attentionによる特徴量とBertLayerへのもとの入力を足し算する層
        self.output = BertOutPut(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states: Embedderモジュールの出力テンソル
        [batch_size, seq_len, hidden_size]
        attention_mask: Transformer のマスクと同じ働きのマスキング
        attention_show_flg: Self-Attention の重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときはattention_probsもリターンする'''
            attention_output, attention_probs = self.attention(
                hidden_states,attention_mask, attention_show_flg
            )
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg
            )
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output

class BertAttention(nn.Module):
    '''BertLayerモジュールのSelf-Attention部分'''
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        '''
        input_tensor: Embeddings モジュールもしくは前段のBertLayerからの出力
        attention_mask: Transformerのマスクと同じ働きのマスキング
        attention_show_flg: self-Attention の重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときはattnetion_probsもリターンする'''
            self_output, attention_probs = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output

class BertSelfAttention(nn.Module):
    '''BertAttentionのSelf-Attention'''
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        # num_attention_heads: 12

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size

        # self-attentionの特徴量を作成する全結合層
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        '''multi-head Attetnion 用にテンソルの形を変換する
        [batch_size, seq_len,hidden] → [batch_size, 12 ,seq_len, hidden/12]
        '''

        new_x_shape = x.size()[x:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def 
