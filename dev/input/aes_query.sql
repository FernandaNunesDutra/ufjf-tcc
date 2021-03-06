SELECT
U.id AS id,
(TIMESTAMPDIFF(YEAR,birth_date,CURDATE())) AS age,
gender = 'F' AS gender,
(IFNULL(audit_1,0) + IFNULL(audit_2,0) + IFNULL(audit_3,0) + IFNULL(audit_4,0) + IFNULL(audit_5,0) + IFNULL(audit_6,0) + IFNULL(audit_7,0) + IFNULL(audit_8,0) + IFNULL(audit_9,0) + IFNULL(audit_10,0)) AS audit,
sunday,
monday,
tuesday,
wednesday,
thursday,
friday,
saturday,
CAST(quit AS UNSIGNED) AS quit,
CAST(dependence_continue AS UNSIGNED) AS dependence_continue
FROM tb_user U INNER JOIN tb_evaluation E ON U.id = E.user_id
WHERE U.id IN (
'579',
'639',
'640',
'649',
'656',
'668',
'674',
'675',
'677',
'682',
'685',
'697',
'712',
'727',
'741',
'746',
'773',
'817',
'852',
'872',
'882',
'905',
'998',
'1047',
'1081',
'1194',
'1308',
'1389',
'1408',
'1424',
'1454',
'1513',
'1528',
'1533',
'1794',
'1874',
'1884',
'1899',
'1909',
'2108',
'2170',
'2197',
'2211',
'2291',
'2318',
'2343',
'2531',
'2675',
'2699',
'2757',
'2775',
'2788',
'2803',
'2838',
'2854',
'2979',
'3037',
'3164',
'3165',
'3234',
'3290',
'3298',
'3373',
'3482',
'3553',
'3599',
'3658',
'3715',
'3776',
'3787',
'3796',
'3904',
'3959',
'4071',
'4087',
'4094',
'4188',
'4268',
'4430',
'4469',
'4530',
'4537',
'4560',
'4609',
'4903',
'4909',
'4958',
'5008',
'5017',
'5091',
'5228',
'5459',
'5648',
'5664',
'5709',
'5712',
'5722',
'5726',
'5745',
'5798',
'5831',
'5837',
'5848',
'6004',
'6013',
'6129',
'6161',
'6177',
'6274',
'6312',
'6340',
'6457',
'6507',
'6730',
'6782',
'6854',
'6884',
'6913',
'6946',
'6955',
'7004',
'7015',
'7041',
'7055',
'7093',
'7152',
'7206',
'7309',
'7363',
'7391',
'7414',
'7441',
'7468',
'7505',
'7522',
'7615',
'7621',
'7649',
'7718',
'7741',
'7745',
'7749',
'7945',
'7961',
'8065',
'8194',
'8238',
'8270',
'8843',
'8863',
'8877',
'8912',
'9165',
'9166');
