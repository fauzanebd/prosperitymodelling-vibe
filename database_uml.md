# Database UML Diagrams

## Overview

This document contains UML diagrams for the database tables in the Kesejahteraan project.

## User Model

```plantuml
@startuml
entity "User" as user {
  * id: Integer <<PK>>
  --
  * username: String(64) <<unique>>
  * password_hash: String(512)
  * is_admin: Boolean
}
@enduml
```

## Indicator Models

```plantuml
@startuml
abstract "BaseIndicator" as base {
  * id: Integer <<PK>>
  --
  * region: String(64) <<index>>
  * year: Integer <<index>>
}

entity "EconomicIndicators" as economic {
  * id: Integer <<PK>>
  --
  * region: String(64) <<index>>
  * year: Integer <<index>>
  * value: Float
  * label_sejahtera: String(64)
  * unit: String
}

entity "HealthIndicators" as health {
  * id: Integer <<PK>>
  --
  * region: String(64) <<index>>
  * year: Integer <<index>>
  * value: Float
  * label_sejahtera: String(64)
  * unit: String
}

entity "EducationIndicators" as education {
  * id: Integer <<PK>>
  --
  * region: String(64) <<index>>
  * year: Integer <<index>>
  * value: Float
  * label_sejahtera: String(64)
  * unit: String
}

entity "InfrastructureIndicators" as infrastructure {
  * id: Integer <<PK>>
  --
  * region: String(64) <<index>>
  * year: Integer <<index>>
  * value: Float
  * label_sejahtera: String(64)
  * unit: String
}

base <|-- economic
base <|-- health
base <|-- education
base <|-- infrastructure

note bottom of economic
  Includes:
  - IndeksPembangunanManusia
  - TingkatPengangguranTerbuka
  - PdrbHargaKonstan
  - PendudukMiskin
  - JmlPengeluaranPerKapita
  - JmlPendudukBekerja
  - DaftarUpahMinimum
endnote

note bottom of health
  Includes:
  - AngkaHarapanHidup
  - FasilitasKesehatan
  - KematianBalita
  - KematianBayi
  - KematianIbu
  - PersentaseBalitaStunting
  - ImunisasiDasar
endnote

note bottom of education
  Includes:
  - AngkaMelekHuruf
  - AngkaPartisipasiKasarSD/SMP/SMA/PT
  - AngkaPartisipasiMurniSD/SMP/SMA/PT
  - RataRataLamaSekolah
endnote

note bottom of infrastructure
  Includes:
  - SanitasiLayak
  - HunianLayak
  - AksesAirMinum
  - KawasanPariwisata
  - KendaraanRoda2/4
  - PanjangRuasJalan
  - TitikLayananInternet
endnote
@enduml
```

## ML Models and Predictions

```plantuml
@startuml
entity "TrainedModel" as trained_model {
  * id: Integer <<PK>>
  --
  * model_type: String(64)
  * created_at: DateTime
  * model_binary: LargeBinary
  * scaler_binary: LargeBinary
  * feature_names: Text
  * accuracy: Float
  * precision: Float
  * recall: Float
  * f1_score: Float
  * test_accuracy: Float
  * training_time: Float
  * inference_time: Float
  * confusion_matrix: Text
  * feature_importance: Text
  * cv_scores: Text
  * mean_cv_accuracy: Float
  * std_cv_accuracy: Float
  * training_parameters: Text
}

entity "RegionPrediction" as prediction {
  * id: Integer <<PK>>
  --
  * region: String(64) <<index>>
  * year: Integer <<index>>
  * model_id: Integer <<FK>>
  * predicted_class: String(64)
  * prediction_probability: Float
  * created_at: DateTime
  --
  * <<UC>> (region, year, model_id)
}

entity "LabelingThreshold" as threshold {
  * id: Integer <<PK>>
  --
  * indicator: String(255) <<unique>>
  * sejahtera_threshold: String(255)
  * menengah_threshold: String(255)
  * tidak_sejahtera_threshold: String(255)
  * labeling_method: String(50)
  * is_reverse: Boolean
  * low_threshold: Float
  * high_threshold: Float
}

trained_model ||--o{ prediction
@enduml
```

## Complete Database Schema

```plantuml
@startuml DatabaseSchema
' Users
entity "User" as user {
  * id: Integer <<PK>>
  --
  * username: String(64) <<unique>>
  * password_hash: String(512)
  * is_admin: Boolean
}

' Base Indicator (Abstract)
abstract "BaseIndicator" as base_indicator {
  * id: Integer <<PK>>
  --
  * region: String(64) <<index>>
  * year: Integer <<index>>
}

' Economic Indicators
entity "IndeksPembangunanManusia" as ipm {
  * id: Integer <<PK>>
  --
  * value: Float
  * label_sejahtera: String(64)
  * unit: String
}

entity "TingkatPengangguranTerbuka" as tpt {
  * id: Integer <<PK>>
  --
  * value: Float
  * label_sejahtera: String(64)
  * unit: String
}

' ML Models
entity "TrainedModel" as trained_model {
  * id: Integer <<PK>>
  --
  * model_type: String(64)
  * created_at: DateTime
  * model_binary: LargeBinary
  * scaler_binary: LargeBinary
  * feature_names: Text
  * accuracy: Float
  * precision: Float
  * recall: Float
  * f1_score: Float
  * training_parameters: Text
}

' Predictions
entity "RegionPrediction" as prediction {
  * id: Integer <<PK>>
  --
  * region: String(64) <<index>>
  * year: Integer <<index>>
  * model_id: Integer <<FK>>
  * predicted_class: String(64)
  * prediction_probability: Float
  * created_at: DateTime
}

' Thresholds
entity "LabelingThreshold" as threshold {
  * id: Integer <<PK>>
  --
  * indicator: String(255) <<unique>>
  * sejahtera_threshold: String(255)
  * menengah_threshold: String(255)
  * tidak_sejahtera_threshold: String(255)
  * labeling_method: String(50)
  * is_reverse: Boolean
  * low_threshold: Float
  * high_threshold: Float
}

' Relationships
base_indicator <|-- ipm
base_indicator <|-- tpt

trained_model ||--o{ prediction
@enduml
```
