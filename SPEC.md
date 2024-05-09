# Access to Everything

* **Green space**
    - Parks, open space, public gardens, organised recreation areas, playgrounds
* **Health and Lifestyle**
    - GPs, pharmacies, dentists, sports facilities, swimming pool, gyms
* **Education**
    - Schools, childcare, universities, day centres
* **Sustenance and essentials**
    - Convenience stores, grocery stores, off-licences, supermarkets, fresh food markets
* **Transport**
    - bus stops, train stations, subway stations, tram stops, bicycle parking, EV charging
* **Community and Culture**
    - Event spaces, places of worship, cinemas, museums
* **Services**
    - Banks, Post offices, beauty salons
* **Food and drink**
    - Restaurants, bars, nightclubs, fast food, cafes
* **Retail**
    - Shopping malls, local shops
* **Employment**
    - workplaces, businesses
* **Civic institutions**
    - city offices, police stations, civic institutions

Notes and potential issues with the current selection;

* Overlap with 'Sustenance' and 'Retail'
* Overlap with 'Green space' and 'Sports Facilities'
* Lack of coherence with 'Services'?

## Overture

The Overture data contains ~2,000 unique categories, which makes classifying them difficult. Can maybe use GPT3.5 with few shot. Need to first decide on the exact categories we want to extract.

From the Overture data we can extract categories relating to;

* **Health and Lifestyle**
* **Sustenence**
* **Community and Culture**
* **Services**
* **Food and Drink**
* **Retail**

## Get Information about Schools

**Education**

* **Schools**
    - e.g. Academy converter, Community school, Free schools etc.
    - **Geography:** Postcodes

**URL**: https://get-information-schools.service.gov.uk/

## National Chargepoint Registry

**Transport**

* **EV Charging:**
    - **Geography:** Lat/Long

**URL:** https://www.gov.uk/guidance/find-and-use-data-on-public-electric-vehicle-chargepoints

## OS Open Greenspace

**Green space**
    - public parks, playing fields, allotments, etc.

**Health and Lifestyle**:
    - sports facilities, play areas

**URL**: https://osdatahub.os.uk/downloads/open/OpenGreenspace

## NHS England

**Health and Lifestyle**

* **Hospitals:** https://files.digital.nhs.uk/assets/ods/current/ets.zip
* **GP Practices**: https://files.digital.nhs.uk/assets/ods/current/epraccur.zip
* **Dentists:** https://files.digital.nhs.uk/assets/ods/current/egdpprac.zip
* **Pharmacies**: https://files.digital.nhs.uk/assets/ods/current/edispensary.zip

* **Query URLs:**  
    - https://digital.nhs.uk/services/organisation-data-service/export-data-files/csv-downloads/other-nhs-organisations
    - https://digital.nhs.uk/services/organisation-data-service/export-data-files/csv-downloads/gp-and-gp-practice-related-data

### NHS Scotland

**Health and Lifestyle**

* **Hospitals:** https://www.opendata.nhs.scot/dataset/hospital-codes
* **GP Practices:** https://www.opendata.nhs.scot/dataset/gp-practice-contact-details-and-list-sizes
* **Dentists**: https://www.opendata.nhs.scot/dataset/dental-practices-and-patient-registrations
* **Pharmacies:** https://www.opendata.nhs.scot/dataset/dispenser-location-contact-details

* **Query URLs:**
    - https://www.opendata.nhs.scot/dataset

## NHS Wales

**Health and Lifestyle**

* **Pharmacies:** https://nwssp.nhs.wales/ourservices/primary-care-services/primary-care-services-documents/pharmacy-practice-dispensing-data-docs/dispensing-data-report-november-2023

* **Query URLs**:
    - https://nwssp.nhs.wales/ourservices/primary-care-services/general-information/data-and-publications

*NOTE:* Typically NHS covers Wales, but this is not the case for pharmacies.
