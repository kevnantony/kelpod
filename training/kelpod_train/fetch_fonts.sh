#!/bin/bash
# Fetch the open-licensed (SIL OFL 1.1) Google Fonts used by style_v2 into ~/kelpod-data/fonts-v2.
# Source: https://github.com/google/fonts (ofl/<family>). Checksums: SHA256SUMS in that directory.
set -u
D=~/kelpod-data/fonts-v2; mkdir -p "$D"; cd "$D"
get() { curl -sfL -m 60 -o "$2" "https://raw.githubusercontent.com/google/fonts/main/ofl/$1" || echo "FAIL $1"; }
get ptsans/PT_Sans-Web-Regular.ttf PT-Sans-Web-Regular.ttf
get lato/Lato-Regular.ttf Lato-Regular.ttf
get barlow/Barlow-Regular.ttf Barlow-Regular.ttf
get ptserif/PT_Serif-Web-Regular.ttf PT-Serif-Web-Regular.ttf
get cardo/Cardo-Regular.ttf Cardo-Regular.ttf
get architectsdaughter/ArchitectsDaughter-Regular.ttf ArchitectsDaughter-Regular.ttf
get kalam/Kalam-Regular.ttf Kalam-Regular.ttf
get patrickhand/PatrickHand-Regular.ttf PatrickHand-Regular.ttf
get shadowsintolight/ShadowsIntoLight.ttf ShadowsIntoLight.ttf
get gloriahallelujah/GloriaHallelujah.ttf GloriaHallelujah.ttf
get reeniebeanie/ReenieBeanie.ttf ReenieBeanie.ttf
get handlee/Handlee-Regular.ttf Handlee-Regular.ttf
get coveredbyyourgrace/CoveredByYourGrace.ttf CoveredByYourGrace.ttf
get karla/Karla%5Bwght%5D.ttf Karla.ttf
get opensans/OpenSans%5Bwdth%2Cwght%5D.ttf OpenSans.ttf
get sourcesans3/SourceSans3%5Bwght%5D.ttf SourceSans3.ttf
get lora/Lora%5Bwght%5D.ttf Lora.ttf
get caveat/Caveat%5Bwght%5D.ttf Caveat.ttf
get ibmplexserif/IBMPlexSerif-Regular.ttf IBMPlexSerif-Regular.ttf
get crimsonpro/CrimsonPro%5Bwght%5D.ttf CrimsonPro.ttf
get gaegu/Gaegu-Regular.ttf Gaegu-Regular.ttf
get rubik/Rubik%5Bwght%5D.ttf Rubik.ttf
get dmsans/DMSans%5Bopsz%2Cwght%5D.ttf DMSans.ttf
get worksans/WorkSans%5Bwght%5D.ttf WorkSans.ttf
get librebaskerville/LibreBaskerville%5Bwght%5D.ttf LibreBaskerville.ttf
get cabin/Cabin%5Bwdth%2Cwght%5D.ttf Cabin.ttf
sha256sum *.ttf > SHA256SUMS
