# BodyMorph Preprocessor

## Auftragsverlauf

Nach „Bilder verarbeiten“ läuft der Auftrag auf dem Server weiter. Der Browser
kann geschlossen werden oder der Rechner in den Standby gehen. Im
Auftragsverlauf können Ergebnisse später als ZIP oder einzelne PNGs
heruntergeladen werden. Aufträge werden nacheinander verarbeitet und können
abgebrochen werden; bereits fertiggestellte Bilder bleiben erhalten.

Ergebnisse werden nicht automatisch gelöscht. Über „Löschen bestätigen“ und
„Auftrag und Ergebnisse löschen“ lässt sich ein beendeter Auftrag einschließlich
seiner Dateien entfernen. Alle Benutzer der Anwendung sehen denselben Verlauf
und können Aufträge herunterladen, abbrechen und löschen. Die Anwendung sollte
deshalb nur in einem vertrauenswürdigen Netzwerk erreichbar sein.

Die Daten liegen unter `/data/output/jobs`. Das in `compose,yml` eingebundene
Verzeichnis `./data` muss dauerhaft erhalten bleiben und ausreichend freien
Speicher bieten. Ein Neustart des Servers unterbricht laufende und wartende
Aufträge; sie werden nicht automatisch fortgesetzt. Bereits gespeicherte Bilder
bleiben im Verlauf sichtbar. Dateien aus älteren Versionen direkt unter
`/data/output` werden nicht automatisch gelöscht oder in den Verlauf übernommen.
