SELECT
U.id AS id,
(TIMESTAMPDIFF(YEAR,birth,CURDATE())) AS age,
gender = 'F' AS gender,
data_parar != data_inserido as data_parar,
CAST(enfrentar_fissura_beber_agua AS UNSIGNED) AS beber_agua,
CAST(enfrentar_fissura_comer AS UNSIGNED) AS comer,
CAST(enfrentar_fissura_ler_razoes AS UNSIGNED) AS ler_razoes,
CAST(enfrentar_fissura_relaxamento AS UNSIGNED) AS relaxamento,
CAST(tentou_parar  AS UNSIGNED) AS tentou_parar,
ladder
FROM tb_user U INNER JOIN tb_pronto_para_parar P ON U.id = P.usuario_id
WHERE U.id IN (
'634',
'805',
'807',
'808',
'817',
'826',
'829',
'830',
'835',
'837',
'838',
'843',
'850',
'853',
'855',
'856',
'866',
'867',
'880',
'881',
'889',
'890',
'894',
'895',
'900',
'906',
'909',
'910',
'913',
'922',
'924',
'926',
'932',
'933',
'942',
'944',
'947',
'950',
'952',
'956',
'958',
'959',
'963',
'965',
'971',
'981',
'984',
'988',
'989',
'990',
'1001',
'1002',
'1005',
'1011',
'1026'
'1032',
'1048',
'1052',
'1055',
'1067',
'1085',
'1108',
'1110',
'1114'
'1120',
'1122',
'1124',
'1126',
'1135',
'1138',
'1143',
'1150',
'1153',
'1154',
'1159',
'1160',
'1170',
'1179',
'1181',
'1188',
'1192',
'1199',
'1201',
'1205',
'1208',
'1217',
'1251',
'1262',
'1273',
'1292',
'1294',
'1296',
'1298',
'1299',
'1301',
'1315',
'1322',
'1325',
'1329',
'1365',
'1378',
'1384',
'1385',
'1419',
'1421',
'1423',
'1438',
'1455',
'1457',
'1460');
