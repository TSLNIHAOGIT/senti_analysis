path='/Users/ozintel/Downloads/self.embedding_size'
import numpy as np
with open(path) as file_emebed:
    # print('file_emebed.shape',)
    for index,each in enumerate(file_emebed):
        line_list=each.strip().split()
        print('line',np.array(line_list[1:]).shape)
        # print('line',[np.float32(each) for each in line_list[1:]])
        print('line {}'.format(index),np.float32(line_list[1:]))
        if index>10:
            break
print([1,2,3,4])

'''
line (1,)
each 352272 300

line (300,)
each ， -0.164651 -0.802285 -0.922292 -0.081340 0.011307 0.819555 0.399610 -0.146892 -0.473265 0.158494 -0.790464 0.464529 0.470437 0.087721 0.514300 -0.286106 -0.262502 -0.023149 -0.637740 -0.119952 0.370561 0.680995 -0.085441 0.103873 -0.881746 0.643512 -0.106018 0.345849 0.416145 -0.295069 -0.316741 -0.359462 0.751065 0.590678 -0.045518 -0.546334 -0.517151 -0.735862 0.000684 -0.346727 -0.737035 -0.249395 0.148803 1.130564 -0.468052 0.308266 0.727099 0.179913 0.122172 0.052392 -0.358235 -0.782753 0.082572 0.080268 0.225968 0.277173 0.216641 0.503959 0.642646 -1.043655 0.777051 -0.264007 -0.030746 -0.416186 0.042599 -0.322579 -0.436427 0.117251 0.147603 -0.015861 0.843740 0.560466 1.189107 -0.927608 0.434663 0.264611 0.479816 -0.255517 -0.669951 -0.727441 -0.601347 0.128198 -0.016777 0.126162 -0.769079 0.191126 0.710765 -0.116890 0.146971 0.134820 0.457911 -0.292842 0.011126 -0.738546 -0.417802 0.183819 -0.222172 -0.053824 -0.548725 -0.122941 0.105433 0.261142 -0.707693 -0.398878 0.307502 0.105698 0.102521 -0.189024 -0.412057 -0.865912 0.214148 0.159483 -0.673333 0.050089 0.580348 1.499760 0.137049 -0.037246 0.451653 -0.232708 -0.766542 -0.322928 0.246333 -0.041459 0.440622 -0.581244 0.191948 -0.056728 0.006118 -0.948995 0.228250 0.059720 -0.189202 0.497123 -0.682191 -0.558279 -0.658866 -0.140244 -0.077984 0.391408 0.299423 0.125846 0.925948 -0.062961 0.117655 0.485773 0.418637 -0.000313 0.059370 0.559151 0.884381 0.089971 0.162024 0.048368 0.357614 -1.202057 -0.317131 -0.478154 -0.180300 0.221121 -0.042714 -0.569975 0.448516 -0.195973 0.254192 0.040981 0.421286 0.643428 -0.588830 -0.809894 -0.615615 0.212922 0.356622 0.141370 0.391299 0.302543 0.760584 -0.571503 0.489856 0.165478 -0.349113 -0.255083 0.332071 -0.438263 1.071083 -0.111300 -0.030014 -0.085093 -0.008353 0.215475 0.920925 -0.082461 -0.203364 0.388918 -0.042622 -0.118885 0.410186 -0.340151 0.099107 0.967955 -0.043266 -0.376376 0.320857 0.093939 0.317655 0.434117 -0.279888 -0.607070 0.488297 -0.838702 0.073795 -0.327615 -0.193701 -0.339342 0.644132 0.036131 0.298061 -0.366357 -0.375713 0.548055 -0.461413 -0.027547 -0.207239 0.059064 0.320076 0.428100 0.411851 0.538310 1.242044 -0.853766 -0.027060 0.440442 -0.459160 0.634165 -0.064710 0.256859 -0.124302 0.221114 0.127552 0.512675 -0.056862 -0.281459 -0.267634 0.285564 -0.712561 0.101558 0.004671 -0.079200 0.467023 -0.375680 0.035121 -0.154638 0.209404 -0.453181 0.391939 0.111594 -0.571840 -0.051020 0.440170 -0.214294 -0.001277 0.010407 -0.950318 0.247196 -0.802545 -0.652084 -0.341805 0.753261 -0.453127 -0.390344 -0.061694 -0.151884 0.630269 -0.341466 0.017448 0.717817 0.157379 0.413284 -0.121750 0.664975 0.223127 -0.358733 -0.802051 -0.477016 -0.155882 -0.433734 -0.385002 0.353909 0.193274 -0.364309 -0.102880 -0.368383 -0.587389 -0.928525 -0.346843 -1.220534 0.007157 -0.400239 0.224750 -0.418401 

line (300,)
each 的 0.650941 0.159002 -0.861428 -0.313324 -1.019395 0.682153 -0.415738 0.051716 -0.085155 0.390126 0.395385 -0.646910 0.211369 0.727402 0.276199 -0.068667 -0.105093 0.582777 0.152467 -0.346212 0.407595 0.008542 0.524375 -0.577848 -0.069541 1.280255 -0.190174 0.594186 0.001650 0.676435 0.107995 -0.179879 1.637642 0.376835 -0.251735 -0.366366 -0.024306 0.204733 0.059973 -0.809230 -0.182821 0.430260 -0.308764 0.482948 0.731772 0.292045 0.060842 0.574854 0.094176 0.377615 -0.716244 -0.079086 -0.916215 0.437628 -0.052980 -0.449339 1.011430 -0.725050 -0.201795 -1.148748 0.854526 -0.333744 0.406687 0.047065 0.160482 -0.692791 -0.147982 0.278991 -0.242640 0.180044 -0.901559 0.569763 -0.426356 -0.110279 0.396697 0.287639 -0.057796 1.104051 -0.346435 0.045485 -0.189843 0.633577 -0.082244 -0.467202 0.025377 -0.311833 -0.094552 -0.404012 0.082546 -0.099592 -0.198990 -0.303303 0.159388 -0.010388 0.328166 0.447528 -0.185682 -0.248173 0.205137 -0.551709 0.023453 -0.970184 -0.715069 -0.272212 -1.271648 -0.301477 -0.331409 0.231878 -0.625196 -0.501584 -0.449269 -0.155813 -0.487931 -0.200711 -0.054340 -0.347144 0.535911 0.399483 -0.119430 0.212451 0.533470 0.730028 -0.889421 0.419797 0.409978 -0.089656 -0.634256 0.214792 0.412098 -0.185526 -0.068842 -0.160234 0.146213 -0.505583 -0.157541 -0.062153 0.996067 0.308703 -0.314910 0.363162 0.160555 0.438329 -0.086266 -0.201213 0.295952 -0.131843 0.411916 -0.039673 -0.577196 0.062966 -0.596115 0.303266 0.373898 0.919810 1.097642 0.475584 -0.492503 0.180445 0.073273 0.132991 -0.461480 0.339120 0.393072 0.517748 -0.300491 -0.055321 -0.525615 0.024775 0.063189 0.547500 -0.078967 0.354487 -0.598455 0.116993 -0.069858 -0.188452 0.542765 0.036732 -0.250532 -0.003326 -0.515640 -0.669085 0.163087 -0.527849 0.253694 -1.014069 -0.758236 0.533713 -0.145347 -0.591781 0.244594 -0.264633 -0.199748 -0.003050 0.200676 0.010386 -0.299595 -0.701712 -0.451363 0.136626 -0.151840 -0.374436 0.457680 -0.017303 0.784021 -0.268131 0.231317 0.321720 0.469805 -0.221154 0.130799 0.443872 0.167399 -0.179009 -1.299259 -1.021607 0.318545 -0.699978 -0.124920 0.228353 -0.755139 -0.015157 -0.075123 -0.122703 0.022181 0.386803 -0.289092 0.039841 0.786477 0.176049 -0.094391 0.358108 0.099875 -0.090206 -0.004977 0.422612 -0.489737 0.132384 0.187841 0.857682 -0.134706 -0.028369 0.630593 0.247921 -1.294616 -0.413146 -0.427982 0.388209 -0.654359 -0.267506 -0.562797 -0.084313 -0.398124 -0.321857 -0.481696 0.310311 0.347906 -0.610156 -0.261714 -0.368281 0.603524 0.491563 -0.258201 0.541466 0.428050 0.703061 -0.810408 0.464519 0.269148 0.956300 -0.063740 -0.148912 0.205212 1.160052 0.877030 0.305806 -0.207060 0.386007 -0.082517 0.461269 -0.427929 -0.842944 0.289134 0.185224 -0.358871 -1.208297 0.117509 0.302484 -0.020503 0.380894 0.287310 -0.057456 -0.235843 0.699682 -0.763621 0.456659 0.179607 -0.075947 -0.039548 0.067626 


'''