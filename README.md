# Touchbiometrics_export
Project Code by Viet Hoang Le, supervisor: Prof. Jiankun Hu. October 2020

Due to different dataset structures and scenarios, 3 classifiers are made - almost 99% similar code
but with some parameters different.

Refer to my project paper, Simply runs each classifier and inputs parameters to verify the experimental results.

Antal_cv_classifier:
    Only asks for train_size and n_bundle and dataset1 or dataset2.
    Also asks for #runs, approx. 15s per run at train_size = 40

Frank_cv_classifier:
    Asks horizontal/vertical, which scenario, train_size and n_bundle
    Asks for #runs, approx.8s per intrasession run (trainsize = 40) or 13s per intersession run (trainsize = 80)


Syed:
    Asks for train_size, n_bundle, stroke orient(horizontal/vertical)-device-posture combinations
    Approx. 20s per run per combination.

Thanks to the datasets available from:
1)	Frank, M., Biedert, R., Ma, E., Martinovic, I. and Song, D., 2012. Touchalytics: On the applicability of touchscreen input as a behavioral biometric for continuous authentication. IEEE transactions on information forensics and security, 8(1), pp.136-148.
2)	Antal, M., Bokor, Z. and Szabó, L.Z., 2015. Information revealed from scrolling interactions on mobile devices. Pattern Recognition Letters, 56, pp.7-13.
3)	Syed, Z.; Helmick, J.; Banerjee, S.; Cukic, B., "Touch Gesture-based Authentication on Mobile Devices: The Effects of User Posture, Device Size, Configuration, and Inter-session Variability," Journal of Systems and Software, 2019 
