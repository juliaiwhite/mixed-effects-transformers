model=$1
dataset=$2
pooling=$3
train_samples=$4
if [ $pooling == 'complete' ] ; then
    python run.py experiment=$model/$dataset.yaml datamodule.train_samples=$train_samples
else
    if [ $dataset == 'movie_dialogue' ] ; then
        feature_values="action adult adventure animation biography comedy crime documentary drama family fantasy film-noir history horror music musical mystery romance sci-fi short sport thriller war western"
    elif [ $dataset == 'reddit' ] ; then
        feature_values="aww todayilearned apple pokemontrades relationship_advice DebateReligion worldnews nba Naruto hiphopheads"
    elif [ $dataset == 'amazon_reviews' ] ; then
        feature_values="video_games pet_products grocery home electronics beauty baby automotive apparel books"
    elif [$dataset == 'c4'] ; then
        feature_values="frontiersin.org chicagotribune.com link.springer.com aljazeera.com instructables.com npr.org dailymail.co.uk csmonitor.com baltimoresun.com city-data.com"
    fi
    for feature_value in $feature_values; do
        if [ $model == 'prefixtuning' ] ; then
            python run.py experiment=$model/$dataset.yaml datamodule.feature_value=$feature_value datamodule.train_samples=$train_samples model.original_tokens=False model.prefix_size=128 model.num_prefixes=1 model.dropout=False
        else
            python run.py experiment=$model/$dataset.yaml datamodule.feature_value=$feature_value datamodule.train_samples=$train_samples
        fi
    done
fi